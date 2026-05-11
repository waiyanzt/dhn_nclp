"""Train DHN for link prediction (multi-seed, scores CSV + summary CSV).

Mirrors Bishwash's RGCN-LP eval contract: 3 seeds, AUC/AP/Hits@{1,3,5}/MRR,
pairwise log-sigmoid loss, val-BCE early stopping (patience 15, max 200 epochs),
no input features (learned nn.Embedding only).

Usage:
    python train_lp.py --config configs/imdb_lp.yaml \\
        --bundle data/preprocessed/IMDB_dhn_lp_md_v1.pt \\
        --out-dir data/results
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dhn.models import DHN
from dhn.utils import get_act_module, get_optimizer


TASK_TARGET_TYPE = {"md": "director", "mg": "genre", "ml": "link"}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_layers_config(layers_config, feat_dim):
    """Replace any indim == -1 with the running output dim (in_dim for layer 0)."""
    out = []
    prev_out = feat_dim
    for layer in layers_config:
        new_layer = {}
        layer_out = 0
        for kernel_name, vals in layer.items():
            indim, outdim, ks = vals
            if indim == -1:
                indim = prev_out
            new_layer[kernel_name] = (indim, outdim, ks)
            layer_out += outdim
        out.append(new_layer)
        prev_out = layer_out
    return out


def move_bundle_to(data, splits, device):
    data = data.to(device)
    data.mapping_index_dict = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in data.mapping_index_dict.items()
    }
    splits = {k: v.to(device) for k, v in splits.items()}
    return data, splits


# ---- Model -----------------------------------------------------------------

class DHN_LP(nn.Module):
    """Encoder: nn.Embedding -> DHN HomConv stack (per-node, no pooling, no fc head).
    Decoder: dot product of node-pair embeddings.
    """

    def __init__(self, num_nodes, in_dim, layers_config, act_module, **act_kwargs):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, in_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.1)
        # DHN with agg=None creates self.fc. Replace it with Identity to expose
        # per-node HomConv embeddings as the encoder's output.
        self.backbone = DHN(
            out_dim=in_dim,            # placeholder; discarded by Identity swap
            layers_config=layers_config,
            act_module=act_module,
            agg=None,
            **act_kwargs,
        )
        self.backbone.fc = nn.Identity()

    def encode(self, graph):
        n = self.emb.num_embeddings
        node_ids = torch.arange(n, device=self.emb.weight.device)
        graph.x = self.emb(node_ids)
        return self.backbone(graph)


def score_pairs(h, pairs):
    return (h[pairs[:, 0]] * h[pairs[:, 1]]).sum(-1)


def score_neg_table(h, movie_ids, neg_targets):
    """movie_ids: (N,), neg_targets: (N, K) -> (N, K) of dot-product scores."""
    K = neg_targets.shape[1]
    movie_expanded = movie_ids.unsqueeze(1).expand(-1, K)
    return (h[movie_expanded] * h[neg_targets]).sum(-1)


# ---- Loss + metrics --------------------------------------------------------

def pairwise_logsigmoid_loss(scores_pos, scores_neg):
    """scores_pos: (N,), scores_neg: (N, K). Higher = better."""
    diffs = scores_pos.unsqueeze(1) - scores_neg
    return -F.logsigmoid(diffs).mean()


def val_bce(scores_pos, scores_neg):
    """1+K-way BCE for early stopping."""
    all_scores = torch.cat([scores_pos.unsqueeze(1), scores_neg], dim=1)
    all_labels = torch.zeros_like(all_scores)
    all_labels[:, 0] = 1
    return F.binary_cross_entropy_with_logits(all_scores, all_labels).item()


def compute_metrics(scores_pos, scores_neg, hits_k=(1, 3, 5), threshold=0.5):
    sp = scores_pos.detach().cpu().numpy()
    sn = scores_neg.detach().cpu().numpy()
    all_scores = np.concatenate([sp, sn.flatten()])
    all_labels = np.concatenate([np.ones(len(sp)), np.zeros(sn.size)])

    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)

    all_probs = 1.0 / (1.0 + np.exp(-all_scores))
    preds = (all_probs > threshold).astype(int)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    accuracy = accuracy_score(all_labels, preds)

    # Ranks: count how many negatives outscore each positive
    ranks = (sn >= sp[:, None]).sum(axis=1) + 1
    mrr = float((1.0 / ranks).mean())
    hits = {f"hits@{k}": float((ranks <= k).mean()) for k in hits_k}

    return {
        "auc": float(auc),
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "mrr": mrr,
        **hits,
    }


@torch.no_grad()
def evaluate_split(model, graph, pos, neg):
    model.eval()
    h = model.encode(graph)
    scores_pos = score_pairs(h, pos)
    scores_neg = score_neg_table(h, pos[:, 0], neg)
    return scores_pos, scores_neg


# ---- One-seed training loop ------------------------------------------------

def run_one_seed(config, bundle_path, seed, device, scores_csv_path, verbose=True):
    set_seed(seed)

    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")
    data = bundle["data"]
    splits = bundle["splits"]
    meta = bundle["meta"]
    offsets = bundle["node_offsets"]

    if "cuda" in device and not torch.cuda.is_available():
        print(f"  [warn] {device} requested but CUDA unavailable; falling back to CPU")
        device = "cpu"

    data, splits = move_bundle_to(data, splits, device)

    num_nodes = meta["num_nodes_total"]
    in_dim = config["model"]["in_dim"]
    layers_config = resolve_layers_config(config["model"]["layers_config"], in_dim)
    activation_kwargs = config["model"]["activation"].get("kwargs", {})

    model = DHN_LP(
        num_nodes=num_nodes,
        in_dim=in_dim,
        layers_config=layers_config,
        act_module=get_act_module(config["model"]["activation"]["name"]),
        **activation_kwargs,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [seed={seed}] trainable params: {n_params}")

    optimizer_fn = get_optimizer(config["training"]["optimizer"]["name"])
    optimizer = optimizer_fn(
        model.parameters(),
        **config["training"]["optimizer"].get("kwargs", {}),
    )

    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    best_val_bce = float("inf")
    best_state = None
    bad_epochs = 0
    best_epoch = 0

    train_pos = splits["train_pos"]
    train_neg = splits["train_neg"]
    val_pos = splits["val_pos"]
    val_neg = splits["val_neg"]
    test_pos = splits["test_pos"]
    test_neg = splits["test_neg"]

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        h = model.encode(data)
        scores_pos = score_pairs(h, train_pos)
        scores_neg = score_neg_table(h, train_pos[:, 0], train_neg)
        loss = pairwise_logsigmoid_loss(scores_pos, scores_neg)
        loss.backward()
        optimizer.step()

        val_sp, val_sn = evaluate_split(model, data, val_pos, val_neg)
        cur_val_bce = val_bce(val_sp, val_sn)

        if cur_val_bce < best_val_bce:
            best_val_bce = cur_val_bce
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            best_epoch = epoch
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"    epoch {epoch:3d}: loss={loss.item():.4f} "
                  f"val_bce={cur_val_bce:.4f} bad={bad_epochs}")

        if bad_epochs >= patience:
            if verbose:
                print(f"  [seed={seed}] early stop at epoch {epoch} "
                      f"(best val_bce={best_val_bce:.4f} at epoch {best_epoch})")
            break

    train_time_s = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    test_sp, test_sn = evaluate_split(model, data, test_pos, test_neg)
    metrics = compute_metrics(
        test_sp, test_sn,
        hits_k=tuple(config["eval"]["hits_k"]),
        threshold=config["eval"].get("threshold", 0.5),
    )
    metrics["seed"] = seed
    metrics["train_time_s"] = train_time_s
    metrics["best_val_bce"] = best_val_bce
    metrics["best_epoch"] = best_epoch

    if verbose:
        print(f"  [seed={seed}] TEST  AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f} "
              f"MRR={metrics['mrr']:.4f} H@1={metrics['hits@1']:.4f} "
              f"H@3={metrics['hits@3']:.4f} H@5={metrics['hits@5']:.4f}")

    if scores_csv_path is not None:
        write_scores_csv(
            scores_csv_path, test_pos, test_neg, test_sp, test_sn,
            offsets, meta["task"],
        )

    return metrics


# ---- CSV writers -----------------------------------------------------------

def write_scores_csv(path, test_pos, test_neg, scores_pos, scores_neg, offsets, task):
    """One row per (movie, candidate, score, label) on the test set.
    Local ids (CMPNN space) so this can be compared directly with Bishwash's RGCN scores CSVs.
    """
    movie_off = offsets["movie"]
    tgt_off = offsets[TASK_TARGET_TYPE[task]]

    sp = scores_pos.detach().cpu().numpy()
    sn = scores_neg.detach().cpu().numpy()
    tp = test_pos.detach().cpu().numpy()
    tn = test_neg.detach().cpu().numpy()
    N, K = tn.shape

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movie_local", "target_local", "score", "label"])
        for i in range(N):
            m_local = int(tp[i, 0] - movie_off)
            t_local = int(tp[i, 1] - tgt_off)
            w.writerow([m_local, t_local, float(sp[i]), 1])
            for k in range(K):
                neg_local = int(tn[i, k] - tgt_off)
                w.writerow([m_local, neg_local, float(sn[i, k]), 0])


def write_summary_csv(path, per_seed_metrics, task, variant):
    """One row per metric: mean and std across seeds (ddof=0 matches lab convention)."""
    if not per_seed_metrics:
        return
    keys = [k for k in per_seed_metrics[0] if k != "seed"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "variant", "metric", "mean", "std", "n_seeds"])
        for k in keys:
            vals = [m[k] for m in per_seed_metrics]
            if not all(isinstance(v, (int, float)) for v in vals):
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=0))
            w.writerow([task, variant, k, mean, std, len(vals)])


# ---- CLI -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DHN for link prediction (multi-seed).")
    p.add_argument("--config", type=str, default="configs/imdb_lp.yaml")
    p.add_argument("--bundle", type=str, required=True,
                   help="Path to DHN-LP bundle .pt (one per (task, variant)).")
    p.add_argument("--out-dir", type=str, default="data/results")
    p.add_argument("--seeds", type=str, default="",
                   help="Override seed list (comma-separated). Default reads from config.")
    p.add_argument("--device", type=str, default="",
                   help="Override device. Default reads from config.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get("device", "cuda:0")
    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds.strip()
        else config.get("seeds", [1566911444, 20241017, 20251017])
    )

    # Peek at bundle meta so output paths are named correctly
    bundle = torch.load(args.bundle, weights_only=False, map_location="cpu")
    task = bundle["meta"]["task"]
    variant = bundle["meta"]["variant"]
    del bundle

    print(f"=== DHN-LP train | task={task} variant={variant} | seeds={seeds} ===")

    per_seed = []
    for seed in seeds:
        scores_path = os.path.join(
            args.out_dir, f"lp_scores_{task}_{variant}_seed{seed}.csv"
        )
        metrics = run_one_seed(config, args.bundle, seed, device, scores_path, verbose=True)
        per_seed.append(metrics)

    summary_path = os.path.join(args.out_dir, f"lp_summary_{task}_{variant}.csv")
    write_summary_csv(summary_path, per_seed, task, variant)

    print(f"\n=== Summary | task={task} variant={variant} (n={len(per_seed)} seeds) ===")
    for k in ("auc", "ap", "mrr", "hits@1", "hits@3", "hits@5", "f1", "accuracy"):
        vals = [m[k] for m in per_seed]
        print(f"  {k:<10s} {np.mean(vals):.4f} ± {np.std(vals, ddof=0):.4f}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
