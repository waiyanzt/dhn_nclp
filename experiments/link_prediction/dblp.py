"""Train DHN for DBLP paper-venue link prediction (multi-seed).

Forked from the IMDb LP runner because DBLP's eval contract differs from IMDb's:
  - negatives are a FLAT POOL (M, 2), not an (N, K) grid;
  - training samples neg_mult negatives per positive each epoch (pointwise
    log-sigmoid loss, matching Bishwash's run_DBLP_pc.py);
  - test negatives are subsampled to <=3 per paper, then ranked per paper;
  - scores CSV is [paper_id, conf_id, score] (local ids) to drop straight into
    scripts.analysis.kendall_tau_across_seeds.

Shared model/encoder/util code is imported from experiments.link_prediction.imdb.

Usage:
    python -m experiments.link_prediction.dblp --config configs/dblp_lp.yaml \\
        --bundle data/preprocessed/DBLP_dhn_lp_pc_v1.pt --out-dir data/results_dblp
"""
import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
import torch
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

from dhn.utils import get_act_module, get_optimizer
from experiments.link_prediction.imdb import (
    DHN_LP,
    move_bundle_to,
    resolve_layers_config,
    score_pairs,
    set_seed,
    synchronize_if_cuda,
    write_summary_csv,
)


def pointwise_loss(pos_scores, neg_scores):
    """Push positives up and negatives down independently (Bishwash's DBLP loss)."""
    return -(F.logsigmoid(pos_scores).mean() + F.logsigmoid(-neg_scores).mean())


def subsample_neg_per_paper(neg, k, seed):
    """Keep <=k negative rows per head (paper). neg: (M, 2) global-id tensor."""
    if k <= 0 or len(neg) == 0:
        return neg
    neg_np = neg.detach().cpu().numpy()
    rng = np.random.RandomState(seed)
    by_paper = defaultdict(list)
    for idx, p in enumerate(neg_np[:, 0]):
        by_paper[int(p)].append(idx)
    keep = []
    for idxs in by_paper.values():
        if len(idxs) > k:
            idxs = rng.choice(idxs, size=k, replace=False).tolist()
        keep.extend(idxs)
    keep.sort()
    return neg[torch.as_tensor(keep, dtype=torch.long, device=neg.device)]


@torch.no_grad()
def evaluate_test(model, data, test_pos, test_neg, offsets, hits_k, threshold):
    model.eval()
    h = model.encode(data)
    pos_scores = score_pairs(h, test_pos)
    neg_scores = score_pairs(h, test_neg)

    sp = pos_scores.detach().cpu().numpy()
    sn = neg_scores.detach().cpu().numpy()
    y_score = np.concatenate([sp, sn])
    y_true = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    y_prob = 1.0 / (1.0 + np.exp(-y_score))
    preds = (y_prob > threshold).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }

    # Per-paper ranking: rank of the true conf among its candidates.
    tp = test_pos.detach().cpu().numpy()
    tn = test_neg.detach().cpu().numpy()
    cand = defaultdict(list)
    for (p, _c), s in zip(tp, sp):
        cand[int(p)].append((float(s), 1))
    for (p, _c), s in zip(tn, sn):
        cand[int(p)].append((float(s), 0))

    rr, hits = [], {k: [] for k in hits_k}
    for items in cand.values():
        items.sort(key=lambda x: x[0], reverse=True)
        true_ranks = [i + 1 for i, (_s, t) in enumerate(items) if t == 1]
        if not true_ranks:
            continue
        r = min(true_ranks)
        rr.append(1.0 / r)
        for k in hits_k:
            hits[k].append(1.0 if r <= k else 0.0)
    metrics["mrr"] = float(np.mean(rr)) if rr else 0.0
    for k in hits_k:
        metrics[f"hits@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0

    return metrics, (pos_scores, neg_scores)


def write_scores_csv(path, test_pos, test_neg, pos_scores, neg_scores, offsets):
    """[paper_id, conf_id, score] in local-id space (matches run_DBLP_pc.py output
    and scripts.analysis.kendall_tau_across_seeds defaults)."""
    poff, coff = offsets["paper"], offsets["conf"]
    tp = test_pos.detach().cpu().numpy()
    tn = test_neg.detach().cpu().numpy()
    sp = pos_scores.detach().cpu().numpy()
    sn = neg_scores.detach().cpu().numpy()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "conf_id", "score"])
        for (p, c), s in zip(tp, sp):
            w.writerow([int(p - poff), int(c - coff), float(s)])
        for (p, c), s in zip(tn, sn):
            w.writerow([int(p - poff), int(c - coff), float(s)])


def run_one_seed(config, bundle_path, seed, device, scores_csv_path, verbose=True):
    total_start = time.perf_counter()
    set_seed(seed)
    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")
    data, splits, meta, offsets = (
        bundle["data"], bundle["splits"], bundle["meta"], bundle["node_offsets"])

    if "cuda" in device and not torch.cuda.is_available():
        print(f"  [warn] {device} requested but CUDA unavailable; using CPU")
        device = "cpu"
    data, splits = move_bundle_to(data, splits, device)

    in_dim = config["model"]["in_dim"]
    layers_config = resolve_layers_config(config["model"]["layers_config"], in_dim)
    act_kwargs = config["model"]["activation"].get("kwargs", {})
    model = DHN_LP(
        num_nodes=meta["num_nodes_total"], in_dim=in_dim, layers_config=layers_config,
        act_module=get_act_module(config["model"]["activation"]["name"]), **act_kwargs,
    ).to(device)

    opt_fn = get_optimizer(config["training"]["optimizer"]["name"])
    optimizer = opt_fn(model.parameters(), **config["training"]["optimizer"].get("kwargs", {}))

    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    neg_mult = config["eval"].get("neg_mult", 3)
    test_neg_per_paper = config["eval"].get("test_neg_per_paper", 3)
    hits_k = tuple(config["eval"]["hits_k"])
    threshold = config["eval"].get("threshold", 0.5)

    train_pos, train_neg = splits["train_pos"], splits["train_neg"]
    val_pos, val_neg = splits["val_pos"], splits["val_neg"]
    test_pos, test_neg = splits["test_pos"], splits["test_neg"]
    # Fixed, deterministic test-negative subsample (aligns scored pairs across
    # models for Kendall-tau, matching subsample_test_neg_per_paper(seed)).
    test_neg = subsample_neg_per_paper(test_neg, test_neg_per_paper, seed)

    rng = np.random.RandomState(seed)
    best_val, best_state, bad, best_epoch = float("inf"), None, 0, 0
    time_to_best_s = 0.0
    synchronize_if_cuda(device)
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        h = model.encode(data)
        pos_scores = score_pairs(h, train_pos)
        n_neg = neg_mult * len(train_pos)
        idx = rng.choice(len(train_neg), size=n_neg, replace=(n_neg > len(train_neg)))
        neg_scores = score_pairs(h, train_neg[torch.as_tensor(idx, dtype=torch.long, device=device)])
        loss = pointwise_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        # Validation loss on the FULL val pool (deterministic early-stop signal).
        model.eval()
        with torch.no_grad():
            hv = model.encode(data)
            v_loss = pointwise_loss(score_pairs(hv, val_pos), score_pairs(hv, val_neg)).item()

        if v_loss < best_val:
            best_val, bad, best_epoch = v_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            synchronize_if_cuda(device)
            time_to_best_s = time.perf_counter() - train_start
        else:
            bad += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"    epoch {epoch:3d}: loss={loss.item():.4f} val_loss={v_loss:.4f} bad={bad}")
        if bad >= patience:
            if verbose:
                print(f"  [seed={seed}] early stop @ epoch {epoch} (best val={best_val:.4f} @ {best_epoch})")
            break

    synchronize_if_cuda(device)
    train_time_s = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    synchronize_if_cuda(device)
    eval_start = time.perf_counter()
    metrics, (psc, nsc) = evaluate_test(model, data, test_pos, test_neg, offsets, hits_k, threshold)
    synchronize_if_cuda(device)
    eval_time_s = time.perf_counter() - eval_start
    metrics.update(
        seed=seed,
        train_time_s=train_time_s,
        eval_time_s=eval_time_s,
        elapsed_time_s=time.perf_counter() - total_start,
        time_to_best_s=time_to_best_s,
        best_val_loss=best_val,
        best_epoch=best_epoch,
    )

    if verbose:
        h3 = metrics.get("hits@3", float("nan"))
        print(f"  [seed={seed}] TEST AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f} "
              f"MRR={metrics['mrr']:.4f} H@1={metrics['hits@1']:.4f} H@3={h3:.4f}")
        print(f"  [seed={seed}] time train={train_time_s:.2f}s eval={eval_time_s:.2f}s "
              f"elapsed={metrics['elapsed_time_s']:.2f}s best@={time_to_best_s:.2f}s")

    if scores_csv_path is not None:
        write_scores_csv(scores_csv_path, test_pos, test_neg, psc, nsc, offsets)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train DHN for DBLP paper-venue LP (multi-seed).")
    ap.add_argument("--config", default="configs/dblp_lp.yaml")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out-dir", default="data/results_dblp")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = args.device or config.get("device", "cuda:0")
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds.strip() else config.get("seeds", [1566911444, 20241017, 20251017]))

    bundle = torch.load(args.bundle, weights_only=False, map_location="cpu")
    task, variant = bundle["meta"]["task"], bundle["meta"]["variant"]
    del bundle
    print(f"=== DHN-LP DBLP train | task={task} variant={variant} | seeds={seeds} ===")

    per_seed = []
    for seed in seeds:
        scores_path = os.path.join(args.out_dir, f"lp_scores_{task}_{variant}_seed{seed}.csv")
        per_seed.append(run_one_seed(config, args.bundle, seed, device, scores_path, verbose=True))

    summary_path = os.path.join(args.out_dir, f"lp_summary_{task}_{variant}.csv")
    write_summary_csv(summary_path, per_seed, task, variant)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
