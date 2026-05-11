# Deep Homomorphism Networks — Node Classification

This repository contains a refactored implementation of **Deep Homomorphism Networks (DHN)**, originally introduced by Hoang NT and Takanori Maehara in their NeurIPS 2024 paper:

> Hoang NT and Takanori Maehara. "Deep Homomorphism Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

The original work targets graph classification. This implementation adapts the DHN architecture for **node classification**, with the goal of benchmarking the method on an in-house graph dataset.

---

## Method Overview

DHN computes graph representations by counting homomorphisms from a set of pattern graphs (cycles and cliques) into the input graph. For each node, the contribution of each homomorphism is computed via learned, pattern-specific transformations and aggregated back onto the node. This gives each node a representation that is aware of the local and global subgraph structure around it, going beyond standard message-passing GNNs in expressive power.

The key components are:

- **`HomConv` (`layers.py`)** — the core convolution operator. For a given pattern (e.g., a cycle of length 4 or a 3-clique), it applies a learned transformation to every homomorphism of that pattern into the graph and scatters the result back to the root node.

- **`DHN` (`models.py`)** — the full model. Stacks multiple `HomConv` layers, each operating over a configurable set of cycle and clique patterns. The original aggregation module for graph-level readout is retained but will be removed or bypassed for node-level prediction.

- **`graph_enumerations.py`** — precomputes cycle and clique homomorphism mappings for a graph using NetworkX. Results are stored as index tensors for efficient batched lookup during forward passes.

- **`HomDataset` / `HomDataLoader` (`datasets.py`)** — handles data loading and preprocessing. Homomorphism mappings are computed once during dataset construction and cached to disk. A custom collate function (`hom_collate`) handles variable-length mapping tensors across batched graphs.

- **`utils.py`** — builder utilities for layers, activations, optimizers, and learning rate schedulers.

---

## IMDB Link Prediction

DHN is also evaluated on **link prediction** over the IMDb heterogeneous graph, alongside the lab's RGCN baseline (Bishwash's `KC_scripts/MAGNN/preprocess_IMDB_rgcn_lp.py` + `run_IMDB_rgcn_lp.py`). To keep the comparison fair, DHN-LP uses the same CMPNN-determined edge splits and the same graph topology rules per variant as the RGCN baseline; the only difference is the model.

### Tasks and variants

| Task | Predicting | Variants | Negatives per positive |
|------|------------|----------|------------------------|
| `md` | movie ↔ director  | v1, v3            | 19 |
| `mg` | movie ↔ genre     | v1, v2, v3, v4    | 2 (3 genres total) |
| `ml` | movie ↔ imdb_link | v1, v2, v3, v4    | 19 |

Variants (v1–v4) reroute Link's neighbours through different node types — see Bishwash's preprocessor for the canonical topology rules; the same rules are mirrored in `preprocess/preprocess_IMDB_dhn_lp.py`.

### Step 1 — Generate CMPNN shared splits (once)

Run Bishwash's three build scripts (already copied to `KC_scripts/CMPNN/`) against the raw movie metadata to produce deterministic 70/10/20 movie-index splits at seed `1566911444`. Outputs land in `data/preprocessed/CMPNN/`.

```bash
mkdir -p data/preprocessed/CMPNN
uv run python KC_scripts/CMPNN/build_IMDB_md_shared_splits.py \
    --csv data/raw/IMDB/movie_metadata.csv \
    --out data/preprocessed/CMPNN/IMDB_md_shared_splits.npz
uv run python KC_scripts/CMPNN/build_IMDB_mg_shared_splits.py \
    --csv data/raw/IMDB/movie_metadata.csv \
    --out data/preprocessed/CMPNN/IMDB_mg_shared_splits.npz
uv run python KC_scripts/CMPNN/build_IMDB_ml_shared_splits.py \
    --csv data/raw/IMDB/movie_metadata.csv \
    --out data/preprocessed/CMPNN/IMDB_ml_shared_splits.npz
```

Expected shapes: train ≈ (2926, 2), val ≈ (418, 2), test ≈ (836, 2).

### Step 2 — Preprocess DHN-LP bundles

`preprocess/preprocess_IMDB_dhn_lp.py` builds the heterograph from `data/raw/IMDB/movie_metadata.csv` using Bishwash's hardcoded v1–v4 topology rules, flattens it to a single homogeneous PyG graph with global node ids, enumerates `{p1, c2, p3}` patterns, and saves a `.pt` bundle per `(task, variant)`. Only **train** target edges enter the graph; val/test are never added, so leakage prevention is by construction.

**Genre nodes are added to the graph for all three tasks** (Bishwash's RGCN runner has the `movie-genre` relation wired up for every task; this matches that intent). For `mg`, genre is the LP target so only train movie-genre edges go in; for `md` and `ml`, genre is auxiliary structural context — every movie carries its genre edge from `labels.npy` regardless of split (no leakage since genre is not the target).

```bash
python preprocess/preprocess_IMDB_dhn_lp.py --task md --variant v1,v3
python preprocess/preprocess_IMDB_dhn_lp.py --task mg --variant v1,v2,v3,v4 --neg-k 2
python preprocess/preprocess_IMDB_dhn_lp.py --task ml --variant v1,v2,v3,v4
```

Each invocation writes `data/preprocessed/IMDB_dhn_lp_<task>_<variant>.pt` containing:

- `data`: PyG `Data(x=None, edge_index, mapping_index_dict, batch, batch_size)` — `x=None` because LP uses learned `nn.Embedding` instead of input features (matches RGCN baseline parity).
- `splits`: `train_pos/val_pos/test_pos` as `(N, 2)` LongTensors of **global** node ids; `train_neg/val_neg/test_neg` as `(N, K)` of global target ids.
- `node_offsets`: dict mapping type name (`movie`, `director`, `actor`, `link`, `genre`) to its starting global id.
- `meta`: task, variant, per-type and total node counts, neg_k, kendall_keys, patterns used.

### Step 3 — Train (multi-seed, 3 seeds per bundle)

`train_lp.py` runs the lab's standard 3-seed protocol (`1566911444, 20241017, 20251017`) per `(task, variant)` and writes both a per-seed scores CSV and an aggregated summary CSV. Eval contract: AUC, AP, Precision/Recall/F1/Accuracy @ 0.5, Hits@{1,3,5}, MRR. Pairwise log-sigmoid loss; val-BCE early stopping with patience 15 and a 200-epoch cap.

```bash
for task in md mg ml; do
  case $task in
    md) variants="v1 v3" ;;
    mg|ml) variants="v1 v2 v3 v4" ;;
  esac
  for v in $variants; do
    python train_lp.py \
      --config configs/imdb_lp.yaml \
      --bundle data/preprocessed/IMDB_dhn_lp_${task}_${v}.pt \
      --out-dir data/results
  done
done
```

Total sweep: 10 `(task, variant)` combinations × 3 seeds = 30 training runs per pattern set.

### Outputs

Under `data/results/` per `(task, variant)`:

| File | Rows | Use |
|------|------|-----|
| `lp_scores_<task>_<variant>_seed<S>.csv` | `(1+K) × N_test` | `[movie_local, target_local, score, label]` in CMPNN local-id space — directly comparable with Bishwash's RGCN scores CSVs (Kendall τ across pattern sets). |
| `lp_summary_<task>_<variant>.csv` | one row per metric | `[task, variant, metric, mean, std, n_seeds]`. Std is computed with `ddof=0` to match the lab convention. |

### Swapping the homomorphism patterns

When evaluating a different pattern set (e.g. the prof's "our version" patterns), update **both**:

1. `preprocess/preprocess_IMDB_dhn_lp.py` — the `PATTERNS` list and the `PATTERN_FNS` dict at the top of the file, then regenerate every bundle.
2. `configs/imdb_lp.yaml` — the kernel names under `model.layers_config`. Each kernel name must appear as a key in the bundle's `mapping_index_dict` (otherwise the kernel receives no counts and silently outputs zeros).

The pattern set and the config kernels must match exactly; bundles are not portable across pattern sets.

---

## Citation

If you use this code or build on the DHN method, please cite the original authors:

```bibtex
@inproceedings{nt2024deephomomorphism,
  title     = {Deep Homomorphism Networks},
  author    = {Hoang NT and Takanori Maehara},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
