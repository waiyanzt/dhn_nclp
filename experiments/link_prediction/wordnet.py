"""Train DHN for WordNet link prediction (multi-seed).

Architectural differences from DBLP/IMDb:
  - Decoder: DistMult (per-relation diagonal W_r).
  - Evaluation: filtered MRR + Hits@{1,3,10} via full entity ranking
    (both head and tail prediction; standard KG-LP protocol).
  - Loss: BCE with balanced corrupt-head/corrupt-tail negative sampling.
  - No input features: nn.Embedding(num_entities, in_dim) only.

Usage:
    python -m experiments.link_prediction.wordnet \\
        --config configs/wordnet_lp.yaml \\
        --bundle data/preprocessed/WordNet3Hop_dhn_lp_no_changes.pt \\
        --out-dir results/v100/wordnet_lp_baseline
"""
import argparse
import csv
import itertools
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dhn.utils import get_act_module, get_optimizer
from experiments.link_prediction.imdb import (
    DHN_LP,
    move_bundle_to,
    resolve_layers_config,
    set_seed,
    synchronize_if_cuda,
    write_summary_csv,
)


# ---- Model ------------------------------------------------------------------

class DHNWordNetLP(nn.Module):
    """DHN encoder with a DistMult decoder for WordNet."""

    def __init__(self, num_entities, num_relations, in_dim, layers_config,
                 act_module, **act_kwargs):
        super().__init__()
        self.num_entities = num_entities
        self.encoder = DHN_LP(
            num_nodes=num_entities,
            in_dim=in_dim,
            layers_config=layers_config,
            act_module=act_module,
            **act_kwargs,
        )
        out_dim = sum(outdim for _, outdim, _ in layers_config[-1].values())
        self.rel_emb = nn.Embedding(num_relations, out_dim)
        nn.init.uniform_(self.rel_emb.weight, -0.5 / out_dim ** 0.5, 0.5 / out_dim ** 0.5)

    def encode(self, data):
        return self.encoder.encode(data)

    def score(self, h_embs, r_ids, t_embs):
        """DistMult: (h * w_r * t).sum(-1). Inputs may be (B, d)."""
        w_r = self.rel_emb(r_ids)
        return (h_embs * w_r * t_embs).sum(-1)

    def score_all_tails(self, h_embs, r_ids, all_embs):
        """Score all entities as tail: returns (B, N_entities)."""
        w_r = self.rel_emb(r_ids)          # (B, d)
        query = h_embs * w_r               # (B, d)
        return query @ all_embs.T          # (B, N)

    def score_all_heads(self, all_embs, r_ids, t_embs):
        """Score all entities as head: returns (B, N_entities)."""
        w_r = self.rel_emb(r_ids)          # (B, d)
        query = t_embs * w_r               # (B, d)
        return query @ all_embs.T          # (B, N)


# ---- Training ---------------------------------------------------------------

def sample_negative_triples(triples, neg_per_pos, rng, num_entities, device):
    """Corrupt one endpoint per negative, choosing heads and tails evenly."""
    h_ids = triples[:, 0].repeat_interleave(neg_per_pos).clone()
    r_ids = triples[:, 1].repeat_interleave(neg_per_pos)
    t_ids = triples[:, 2].repeat_interleave(neg_per_pos).clone()
    n_neg = len(h_ids)

    replacements = torch.from_numpy(
        rng.randint(0, num_entities, n_neg).astype(np.int64)
    ).to(device)
    corrupt_head = torch.from_numpy(
        rng.randint(0, 2, n_neg).astype(np.bool_)
    ).to(device)
    h_ids[corrupt_head] = replacements[corrupt_head]
    t_ids[~corrupt_head] = replacements[~corrupt_head]
    return h_ids, r_ids, t_ids


def train_epoch(model, data, train_triples, neg_per_pos, rng, device, optimizer):
    model.train()
    optimizer.zero_grad()
    h = model.encode(data)

    h_ids = train_triples[:, 0]
    r_ids = train_triples[:, 1]
    t_ids = train_triples[:, 2]

    pos_scores = model.score(h[h_ids], r_ids, h[t_ids])

    neg_h, neg_r, neg_t = sample_negative_triples(
        train_triples, neg_per_pos, rng, model.num_entities, device
    )
    neg_scores = model.score(h[neg_h], neg_r, h[neg_t])

    loss = -(F.logsigmoid(pos_scores).mean() + F.logsigmoid(-neg_scores).mean())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def compute_val_loss(model, data, val_triples, neg_per_pos, rng, device):
    model.eval()
    h = model.encode(data)
    h_ids = val_triples[:, 0]
    r_ids = val_triples[:, 1]
    t_ids = val_triples[:, 2]
    pos_scores = model.score(h[h_ids], r_ids, h[t_ids])
    neg_h, neg_r, neg_t = sample_negative_triples(
        val_triples, neg_per_pos, rng, model.num_entities, device
    )
    neg_scores = model.score(h[neg_h], neg_r, h[neg_t])
    return -(F.logsigmoid(pos_scores).mean() + F.logsigmoid(-neg_scores).mean()).item()


# ---- Evaluation -------------------------------------------------------------

def build_filter_dicts(all_triples_np):
    """(h,r)->set(known_t), (r,t)->set(known_h) from all splits."""
    filter_tails = defaultdict(set)
    filter_heads = defaultdict(set)
    for h, r, t in all_triples_np:
        filter_tails[(int(h), int(r))].add(int(t))
        filter_heads[(int(r), int(t))].add(int(h))
    return filter_tails, filter_heads


def _ranks_from_scores(scores_np, true_ids_np):
    """For each row, rank of true entity among all (higher score = better)."""
    ranks = np.zeros(len(true_ids_np), dtype=np.int64)
    for i, t in enumerate(true_ids_np):
        ranks[i] = int((scores_np[i] > scores_np[i, t]).sum()) + 1
    return ranks


def build_kendall_candidates(test_pos, val_pos, num_entities, num_candidates,
                             rng_seed):
    """Build the lab-standard shared tail candidates for Kendall analysis."""
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")

    tail_filters = defaultdict(set)
    for triples in (val_pos, test_pos):
        for head, relation, tail in triples.tolist():
            tail_filters[(int(head), int(relation))].add(int(tail))

    candidates = np.empty((len(test_pos), num_candidates), dtype=np.int64)
    for query_id, (head, relation, true_tail) in enumerate(test_pos.tolist()):
        rng = np.random.default_rng(rng_seed + query_id)
        forbidden = set(tail_filters[(int(head), int(relation))])
        forbidden.add(int(true_tail))
        negatives = []
        attempts = 0
        while len(negatives) < num_candidates - 1 and attempts < 64:
            need = (num_candidates - 1 - len(negatives)) * 4 + 8
            sampled = rng.integers(0, num_entities, size=need)
            for candidate in sampled.tolist():
                if candidate in forbidden:
                    continue
                negatives.append(candidate)
                forbidden.add(candidate)
                if len(negatives) == num_candidates - 1:
                    break
            attempts += 1
        if len(negatives) < num_candidates - 1:
            available = (
                entity for entity in range(num_entities)
                if entity not in forbidden
            )
            negatives.extend(
                list(itertools.islice(
                    available, num_candidates - 1 - len(negatives)
                ))
            )
        candidates[query_id, :-1] = negatives
        candidates[query_id, -1] = int(true_tail)
    return candidates


@torch.no_grad()
def score_kendall_candidates(model, data, test_pos, candidates, batch_size,
                             device):
    """Score a fixed tail-candidate matrix with the final checkpoint."""
    model.eval()
    entity_embeddings = model.encode(data)
    scores = np.empty(candidates.shape, dtype=np.float32)

    for start in range(0, len(test_pos), batch_size):
        end = min(start + batch_size, len(test_pos))
        triples = torch.from_numpy(test_pos[start:end]).long().to(device)
        candidate_ids = torch.from_numpy(candidates[start:end]).long().to(device)
        head_embeddings = entity_embeddings[triples[:, 0]]
        relation_embeddings = model.rel_emb(triples[:, 1])
        candidate_embeddings = entity_embeddings[candidate_ids]
        batch_scores = (
            (head_embeddings * relation_embeddings).unsqueeze(1)
            * candidate_embeddings
        ).sum(dim=-1)
        scores[start:end] = batch_scores.cpu().numpy()
    return scores


@torch.no_grad()
def evaluate_filtered(model, data, test_triples, all_triples, hits_k, eval_batch_size,
                       device, binary_k=None, binary_rng_seed=42, verbose=False):
    model.eval()
    h = model.encode(data)                   # (N, d)

    filter_tails, filter_heads = build_filter_dicts(all_triples.cpu().numpy())

    tail_rr, head_rr = [], []
    tail_hits = {k: [] for k in hits_k}
    head_hits = {k: [] for k in hits_k}
    binary_logits, binary_labels = [], []
    binary_rng = np.random.default_rng(binary_rng_seed)

    test_np = test_triples.cpu().numpy()
    n_test = len(test_np)

    for start in range(0, n_test, eval_batch_size):
        batch_np = test_np[start:start + eval_batch_size]
        B = len(batch_np)
        h_ids = torch.from_numpy(batch_np[:, 0]).long().to(device)
        r_ids = torch.from_numpy(batch_np[:, 1]).long().to(device)
        t_ids = torch.from_numpy(batch_np[:, 2]).long().to(device)

        # --- Tail prediction ---
        scores_tail = model.score_all_tails(h[h_ids], r_ids, h).cpu().numpy()  # (B, N)
        for i in range(B):
            hi, ri, ti = int(batch_np[i, 0]), int(batch_np[i, 1]), int(batch_np[i, 2])
            if binary_k:
                known_tails = filter_tails[(hi, ri)]
                negatives = []
                candidates = binary_rng.integers(
                    0, model.num_entities, size=binary_k * 4
                )
                for candidate in candidates:
                    if candidate not in known_tails and len(negatives) < binary_k:
                        negatives.append(int(candidate))
                while len(negatives) < binary_k:
                    negatives.append(
                        int(binary_rng.integers(0, model.num_entities))
                    )

                binary_logits.append(float(scores_tail[i, ti]))
                binary_logits.extend(scores_tail[i, negatives].tolist())
                binary_labels.append(1)
                binary_labels.extend([0] * binary_k)

            for k in filter_tails[(hi, ri)] - {ti}:
                scores_tail[i, k] = -1e9
        ranks_tail = _ranks_from_scores(scores_tail, batch_np[:, 2])
        tail_rr.extend((1.0 / ranks_tail).tolist())
        for k in hits_k:
            tail_hits[k].extend((ranks_tail <= k).tolist())

        # --- Head prediction ---
        scores_head = model.score_all_heads(h, r_ids, h[t_ids]).cpu().numpy()  # (B, N)
        for i in range(B):
            hi, ri, ti = int(batch_np[i, 0]), int(batch_np[i, 1]), int(batch_np[i, 2])
            for k in filter_heads[(ri, ti)] - {hi}:
                scores_head[i, k] = -1e9
        ranks_head = _ranks_from_scores(scores_head, batch_np[:, 0])
        head_rr.extend((1.0 / ranks_head).tolist())
        for k in hits_k:
            head_hits[k].extend((ranks_head <= k).tolist())

        if verbose and (start // eval_batch_size) % 20 == 0:
            pct = 100 * min(start + eval_batch_size, n_test) / n_test
            print(f"    eval {pct:5.1f}%", flush=True)

    mrr_tail = float(np.mean(tail_rr))
    mrr_head = float(np.mean(head_rr))
    metrics = {
        "mrr": (mrr_tail + mrr_head) / 2.0,
        "mrr_tail": mrr_tail,
        "mrr_head": mrr_head,
    }
    for k in hits_k:
        ht = float(np.mean(tail_hits[k]))
        hh = float(np.mean(head_hits[k]))
        metrics[f"hits@{k}"] = (ht + hh) / 2.0
        metrics[f"hits@{k}_tail"] = ht
        metrics[f"hits@{k}_head"] = hh
    if binary_k:
        labels = np.asarray(binary_labels, dtype=np.int32)
        predictions = (np.asarray(binary_logits, dtype=np.float32) > 0).astype(
            np.int32
        )
        metrics.update(
            accuracy=float(accuracy_score(labels, predictions)),
            precision=float(
                precision_score(
                    labels, predictions, average="macro", zero_division=0
                )
            ),
            recall=float(
                recall_score(labels, predictions, average="macro", zero_division=0)
            ),
            f1=float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
        )
    return metrics


# ---- Per-seed runner --------------------------------------------------------

def run_one_seed(config, bundle_path, seed, device, out_dir, verbose=True):
    total_start = time.perf_counter()
    set_seed(seed)
    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")
    meta = bundle["meta"]
    splits = bundle["splits"]

    if "cuda" in device and not torch.cuda.is_available():
        print(f"  [warn] {device} requested but CUDA unavailable; falling back to cpu")
        device = "cpu"

    # Move graph data (not triples) to device
    data = bundle["data"].to(device)
    data.mapping_index_dict = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in data.mapping_index_dict.items()
    }

    train_triples = splits["train"].to(device)
    val_triples = splits["val"].to(device)
    test_triples = splits["test"]       # stays on CPU; eval moves batches
    all_triples = splits["all_triples"] # stays on CPU for filter dict

    in_dim = config["model"]["in_dim"]
    layers_config = resolve_layers_config(config["model"]["layers_config"], in_dim)
    act_kwargs = config["model"]["activation"].get("kwargs", {})

    model = DHNWordNetLP(
        num_entities=meta["num_entities"],
        num_relations=meta["num_relations"],
        in_dim=in_dim,
        layers_config=layers_config,
        act_module=get_act_module(config["model"]["activation"]["name"]),
        mapping_chunk_size=config["model"].get("mapping_chunk_size"),
        checkpoint_chunks=config["model"].get("checkpoint_chunks", True),
        **act_kwargs,
    ).to(device)

    opt_fn = get_optimizer(config["training"]["optimizer"]["name"])
    optimizer = opt_fn(model.parameters(), **config["training"]["optimizer"].get("kwargs", {}))

    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    neg_per_pos = config["training"].get("neg_per_pos", 10)
    hits_k = tuple(config["eval"]["hits_k"])
    eval_batch_size = config["eval"].get("eval_batch_size", 256)
    binary_k = config["eval"].get("binary_k")
    binary_rng_seed = config["eval"].get("binary_rng_seed", 42)

    rng = np.random.RandomState(seed)
    best_val, best_state, bad, best_epoch = float("inf"), None, 0, 0
    time_to_best_s = 0.0
    synchronize_if_cuda(device)
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, train_triples, neg_per_pos, rng, device, optimizer)
        v_loss = compute_val_loss(model, data, val_triples, neg_per_pos,
                                  np.random.RandomState(epoch), device)

        if v_loss < best_val:
            best_val, bad, best_epoch = v_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            synchronize_if_cuda(device)
            time_to_best_s = time.perf_counter() - train_start
        else:
            bad += 1

        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"    epoch {epoch:3d}: loss={loss:.4f} val_loss={v_loss:.4f} bad={bad}")
        if bad >= patience:
            if verbose:
                print(f"  [seed={seed}] early stop @ epoch {epoch} (best @ {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    synchronize_if_cuda(device)
    train_time_s = time.perf_counter() - train_start
    epochs_trained = epoch

    print(f"  [seed={seed}] Running test evaluation (filtered MRR)...", flush=True)
    synchronize_if_cuda(device)
    eval_start = time.perf_counter()
    metrics = evaluate_filtered(
        model, data, test_triples, all_triples, hits_k, eval_batch_size, device,
        binary_k=binary_k, binary_rng_seed=binary_rng_seed, verbose=verbose,
    )
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
        epochs_trained=epochs_trained,
    )

    artifact_config = config.get("artifacts", {})
    dataset_slug = meta.get("dataset_slug", "wordnet")
    if artifact_config.get("save_best_checkpoint", False):
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"best_model_{dataset_slug}_seed{seed}.pt"
        )
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            checkpoint_path,
        )
        print(f"  [seed={seed}] saved checkpoint -> {checkpoint_path}")

    kendall_config = artifact_config.get("kendall", {})
    if kendall_config.get("enabled", False):
        num_candidates = int(kendall_config.get("num_candidates", 200))
        candidate_seed = int(kendall_config.get("candidate_rng_seed", 42))
        test_np = test_triples.numpy()
        val_np = splits["val"].cpu().numpy()
        candidates = build_kendall_candidates(
            test_np,
            val_np,
            meta["num_entities"],
            num_candidates,
            candidate_seed,
        )
        scores = score_kendall_candidates(
            model, data, test_np, candidates, eval_batch_size, device
        )
        labels = np.zeros(candidates.shape, dtype=np.uint8)
        labels[:, -1] = 1
        kendall_path = os.path.join(
            out_dir, f"kendall_tail_scores_{dataset_slug}_seed{seed}.npz"
        )
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            kendall_path,
            test_pos=test_np,
            candidate_ids=candidates,
            scores=scores,
            labels=labels,
            candidate_rng_seed=np.asarray(candidate_seed),
        )
        print(f"  [seed={seed}] saved Kendall scores -> {kendall_path}")

    h1 = metrics.get("hits@1", float("nan"))
    h10 = metrics.get("hits@10", float("nan"))
    print(f"  [seed={seed}] TEST MRR={metrics['mrr']:.4f} MRR_tail={metrics['mrr_tail']:.4f} "
          f"H@1={h1:.4f} H@10={h10:.4f}")
    if binary_k:
        print(
            f"  [seed={seed}] binary(k={binary_k}) "
            f"Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} "
            f"Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )
    print(f"  [seed={seed}] time train={train_time_s:.2f}s eval={eval_time_s:.2f}s "
          f"elapsed={metrics['elapsed_time_s']:.2f}s best@={time_to_best_s:.2f}s "
          f"epochs={epochs_trained}")
    return metrics


# ---- Summary CSV (KG-LP columns) --------------------------------------------

def write_kg_summary_csv(path, per_seed, hits_k):
    if not per_seed:
        return
    keys = (["seed", "mrr", "mrr_tail", "mrr_head"]
            + [f"hits@{k}" for k in hits_k]
            + [f"hits@{k}_tail" for k in hits_k]
            + [f"hits@{k}_head" for k in hits_k]
            + ["accuracy", "precision", "recall", "f1"]
            + [
                "best_epoch",
                "best_val_loss",
                "train_time_s",
                "eval_time_s",
                "elapsed_time_s",
                "time_to_best_s",
                "epochs_trained",
            ])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    summary_stats = {}

    def mean_std(k):
        vals = [m.get(k, float("nan")) for m in per_seed
                if isinstance(m.get(k), (int, float))]
        return (float(np.mean(vals)), float(np.std(vals, ddof=0))) if vals else (float("nan"), float("nan"))

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["_type"])
        w.writeheader()
        for m in per_seed:
            row = {k: m.get(k, "") for k in keys}
            row["_type"] = "per_seed"
            w.writerow(row)

        mean_row = {"_type": "mean"}
        std_row = {"_type": "std"}
        for k in keys:
            if k == "seed":
                mean_row[k] = ""
                std_row[k] = ""
            else:
                mu, sigma = mean_std(k)
                summary_stats[k] = (mu, sigma)
                mean_row[k] = f"{mu:.6f}"
                std_row[k] = f"{sigma:.6f}"
        w.writerow(mean_row)
        w.writerow(std_row)

    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    raw_path = os.path.join(directory, filename.replace("lp_summary_", "lp_raw_", 1))
    table_path = os.path.join(
        directory, filename.replace("lp_summary_", "lp_table_", 1)
    )

    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for metrics in per_seed:
            writer.writerow({key: metrics.get(key, "") for key in keys})

    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "mean", "std", "n_seeds"]
        )
        writer.writeheader()
        for key in keys:
            if key == "seed":
                continue
            mean, std = summary_stats[key]
            writer.writerow(
                {
                    "metric": key,
                    "mean": f"{mean:.6f}",
                    "std": f"{std:.6f}",
                    "n_seeds": len(per_seed),
                }
            )

    print(f"Wrote {path}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {table_path}")


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train DHN for WordNet link prediction.")
    ap.add_argument("--config", default="configs/wordnet_lp.yaml")
    ap.add_argument(
        "--bundle",
        default="data/preprocessed/WordNet3Hop_dhn_lp_no_changes.pt",
    )
    ap.add_argument("--out-dir", default="results/v100/wordnet_lp_baseline")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config.get("device", "cuda:0")
    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds.strip()
        else config.get("seeds", [1566911444, 20241017, 20251017])
    )

    bundle = torch.load(args.bundle, weights_only=False, map_location="cpu")
    meta = bundle["meta"]
    del bundle
    dataset_name = meta.get("source", meta.get("dataset", "WordNet"))
    dataset_slug = meta.get("dataset_slug", "wordnet")
    print(f"=== DHN-LP {dataset_name} | entities={meta['num_entities']:,} "
          f"relations={meta['num_relations']} | seeds={seeds} ===")

    hits_k = tuple(config["eval"]["hits_k"])
    per_seed = []
    for seed in seeds:
        per_seed.append(run_one_seed(config, args.bundle, seed, device, args.out_dir, verbose=True))

    summary_path = os.path.join(args.out_dir, f"lp_summary_{dataset_slug}.csv")
    write_kg_summary_csv(summary_path, per_seed, hits_k)

    # Print final mean±std
    for key in ["mrr", "hits@1", "hits@10"]:
        vals = [m[key] for m in per_seed if key in m]
        if vals:
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals, ddof=0):.4f}")


if __name__ == "__main__":
    main()
