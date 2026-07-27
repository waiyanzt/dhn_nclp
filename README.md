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

## Repository Layout

- `dhn/` — reusable DHN implementation: layers, models, graph enumeration, datasets, and utility builders.
- `experiments/` — runnable training and benchmark entrypoints, grouped by task.
  - `experiments/node_classification/` — IMDb and Freebase node-classification training and benchmarking.
  - `experiments/link_prediction/` — IMDb, DBLP, and WordNet link-prediction runners.
  - `experiments/original_graph_classification/` — original DHN graph-classification runner.
- `preprocess/` — dataset-specific preprocessing entrypoints, grouped by dataset and task.
- `scripts/diagnostics/` — one-off inspection and enumeration-cost probes.
- `scripts/reports/` — result aggregation and HTML report generation.
- `scripts/analysis/` — score-comparison analysis such as Kendall tau.
- `KC_scripts/` — teammate baseline scripts, left in their original layout.
- `configs/` and `data/` — experiment configs and raw/preprocessed artifacts.
- `results/` — benchmark outputs grouped first by GPU, then dataset and task.

Use module paths from the repo root, for example:

```bash
python -m preprocess.imdb.link_prediction --task md --variant v1,v3
python -m experiments.link_prediction.imdb --config configs/imdb_lp.yaml \
    --bundle data/preprocessed/IMDB_dhn_lp_md_v1.pt \
    --out-dir results/v100/imdb_lp_baseline
```

---

## IMDB Link Prediction

DHN is also evaluated on **link prediction** over the IMDb heterogeneous graph, alongside the lab's RGCN baseline (Bishwash's `KC_scripts/MAGNN/preprocess_IMDB_rgcn_lp.py` + `run_IMDB_rgcn_lp.py`). To keep the comparison fair, DHN-LP uses the same CMPNN-determined edge splits and the same graph topology rules per variant as the RGCN baseline; the only difference is the model.

### Tasks and variants

| Task | Predicting | Variants | Negatives per positive |
|------|------------|----------|------------------------|
| `md` | movie ↔ director  | v1, v3            | 19 |
| `mg` | movie ↔ genre     | v1, v2, v3, v4    | 2 (3 genres total) |
| `ml` | movie ↔ imdb_link | v1, v2, v3, v4    | 19 |

Variants (v1–v4) reroute Link's neighbours through different node types — see Bishwash's preprocessor for the canonical topology rules; the same rules are mirrored in `preprocess/imdb/link_prediction.py`.

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

`preprocess/imdb/link_prediction.py` builds the heterograph from `data/raw/IMDB/movie_metadata.csv` using Bishwash's hardcoded v1–v4 topology rules, flattens it to a single homogeneous PyG graph with global node ids, enumerates `{p1, c2}` patterns, and saves a `.pt` bundle per `(task, variant)`. Only **train** target edges enter the graph; val/test are never added, so leakage prevention is by construction.

**Genre nodes are added to the graph for all three tasks** (Bishwash's RGCN runner has the `movie-genre` relation wired up for every task; this matches that intent). For `mg`, genre is the LP target so only train movie-genre edges go in; for `md` and `ml`, genre is auxiliary structural context — every movie carries its genre edge from `labels.npy` regardless of split (no leakage since genre is not the target).

```bash
python -m preprocess.imdb.link_prediction --task md --variant v1,v3
python -m preprocess.imdb.link_prediction --task mg --variant v1,v2,v3,v4 --neg-k 2
python -m preprocess.imdb.link_prediction --task ml --variant v1,v2,v3,v4
```

Each invocation writes `data/preprocessed/IMDB_dhn_lp_<task>_<variant>.pt` containing:

- `data`: PyG `Data(x=None, edge_index, mapping_index_dict, batch, batch_size)` — `x=None` because LP uses learned `nn.Embedding` instead of input features (matches RGCN baseline parity).
- `splits`: `train_pos/val_pos/test_pos` as `(N, 2)` LongTensors of **global** node ids; `train_neg/val_neg/test_neg` as `(N, K)` of global target ids.
- `node_offsets`: dict mapping type name (`movie`, `director`, `actor`, `link`, `genre`) to its starting global id.
- `meta`: task, variant, per-type and total node counts, neg_k, kendall_keys, patterns used.

### Step 3 — Train (multi-seed, 3 seeds per bundle)

`experiments/link_prediction/imdb.py` runs the lab's standard 3-seed protocol (`1566911444, 20241017, 20251017`) per `(task, variant)` and writes both a per-seed scores CSV and an aggregated summary CSV. Eval contract: AUC, AP, Precision/Recall/F1/Accuracy @ 0.5, Hits@{1,3,5}, MRR. Pairwise log-sigmoid training and validation loss; early stopping uses patience 15 and a 200-epoch cap.

Timing contract for new benchmark runs:

- `train_time_s`: wall-clock training-loop time, synchronized on CUDA so GPU kernels are included.
- `eval_time_s`: final held-out test evaluation time. IMDb, DBLP, and WordNet LP report this separately; IMDb NC keeps this at `0.0` because validation/test evaluation happens inside each epoch.
- `elapsed_time_s`: end-to-end per-seed runner time, including bundle load, setup, training, and final evaluation.
- `time_to_best_s`: wall-clock time from training start until the best validation checkpoint was first observed. This is the closest “time to accuracy/convergence” metric in these scripts.

For lab comparisons, use `train_time_s` plus final metrics as the headline efficiency measure. Use `elapsed_time_s` when comparing full pipeline cost, and do not include preprocessing/enumeration time unless the experiment explicitly studies preprocessing cost.

Standalone DHN node-classification and link-prediction runs also append the
same resource fields used by the augmentation experiments. Their raw and
summary outputs include model/checkpoint bytes, peak process RSS, and separate
training/inference CUDA allocated, reserved, peak-allocated, and peak-reserved
bytes. These are PyTorch allocator measurements, not whole-device utilization
percentages. Rerun cached node-classification seeds to populate these fields;
artifacts created by older code cannot recover historical peak memory.

```bash
for task in md mg ml; do
  case $task in
    md) variants="v1 v3" ;;
    mg|ml) variants="v1 v2 v3 v4" ;;
  esac
  for v in $variants; do
    python -m experiments.link_prediction.imdb \
      --config configs/imdb_lp.yaml \
      --bundle data/preprocessed/IMDB_dhn_lp_${task}_${v}.pt \
      --out-dir results/v100/imdb_lp_baseline
  done
done
```

Total sweep: 10 `(task, variant)` combinations × 3 seeds = 30 training runs per pattern set.

### Outputs

Under `results/v100/imdb_lp_baseline/` per `(task, variant)`:

| File | Rows | Use |
|------|------|-----|
| `lp_scores_<task>_<variant>_seed<S>.csv` | `(1+K) × N_test` | `[movie_local, target_local, score, label]` in CMPNN local-id space — directly comparable with Bishwash's RGCN scores CSVs (Kendall τ across pattern sets). |
| `lp_summary_<task>_<variant>.csv` | one row per metric | `[task, variant, metric, mean, std, n_seeds]`, including timing metrics. Std is computed with `ddof=0` to match the lab convention. |

### Joint IMDb link-prediction data augmentation

`experiments/link_prediction/imdb_augmentation.py` trains one shared DHN,
optimizer, and checkpoint across all valid graph variants for one task. The
task sets match the RGCN augmentation experiment: `md` uses `v1,v3`; `mg` and
`ml` use `v1,v2,v3,v4`. A super-epoch visits every selected variant in seeded
random order, then selects the shared checkpoint by mean validation pairwise
log-sigmoid loss.

Run preflight first to prove that every variant has byte-identical positive and
negative split tables, node spaces, offsets, and pattern keys:

```bash
for task in md mg ml; do
  python -m experiments.link_prediction.imdb_augmentation \
    --task "$task" \
    --config configs/imdb_lp.yaml \
    --preflight-only
done
```

Then run all three tasks:

```bash
for task in md mg ml; do
  python -m experiments.link_prediction.imdb_augmentation \
    --task "$task" \
    --config configs/imdb_lp.yaml \
    --output-dir results/dhn_augmentation/IMDB_LP
done
```

The default `--batch-size 0` retains DHN's existing full-split update. An
explicit smaller value (for example `--batch-size 256`) batches positive rows
and their fixed negatives. This does not change the DHN encoder or decoder, but
it does increase the number of optimizer updates per super-epoch and is
therefore recorded in every summary.

Each task writes `seed_summary.csv` under
`results/dhn_augmentation/IMDB_LP/<task>/`, plus per-seed histories, shared
checkpoints, exact resume state, per-variant test scores and metrics, pairwise
invariance, peak process RSS, and CUDA allocated/reserved/peak memory. Resume an
interrupted task with the identical command plus `--resume`; only
`--super-epochs` and `--device` may change.

### Swapping the homomorphism patterns

When evaluating a different pattern set (e.g. the prof's "our version" patterns), update **both**:

1. `preprocess/imdb/link_prediction.py` — the `PATTERNS` list and the `PATTERN_FNS` dict at the top of the file, then regenerate every bundle.
2. `configs/imdb_lp.yaml` — the kernel names under `model.layers_config`. Each kernel name must appear as a key in the bundle's `mapping_index_dict` (otherwise the kernel receives no counts and silently outputs zeros).

The pattern set and the config kernels must match exactly; bundles are not portable across pattern sets.

---

## DBLP Invariant Link Prediction

DBLP uses a two-layer `{p1, c2}` DHN. This avoids explicitly materializing all
three-node-path homomorphisms and permits preprocessing the full eligible paper
set. First generate the shared paper-disjoint 70/10/20 split:

```bash
python -m preprocess.dblp.shared_splits
```

The three baseline bundles retain one area attachment relation each:

```text
v1: Paper-Area
v2: Venue-Area
v3: Author-Area
```

Figure 3 in the lab paper defines DBLP* as the union graph containing all three
area relation families, together with the common Paper-Author, Paper-Term, and
train-only Paper-Venue edges. Build all baseline and invariant bundles with:

```bash
python -m preprocess.dblp.link_prediction \
  --variant v1,v2,v3,universal \
  --raw-dir data/raw/DBLP \
  --shared-npz data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz \
  --out-dir data/preprocessed
```

This writes one bundle per variant. Train the three baseline bundles into
`results/v100/dblp_lp_baseline` and the universal bundle with the same model,
split, seeds, loss, and hyperparameters:

```bash
for variant in v1 v2 v3; do
  python -m experiments.link_prediction.dblp \
    --config configs/dblp_lp.yaml \
    --bundle "data/preprocessed/DBLP_dhn_lp_pc_${variant}.pt" \
    --out-dir results/v100/dblp_lp_baseline
done

python -m experiments.link_prediction.dblp \
  --config configs/dblp_lp.yaml \
  --bundle data/preprocessed/DBLP_dhn_lp_pc_universal.pt \
  --out-dir results/v100/dblp_lp_invariant
```

The universal bundle changes only the graph-derived `mapping_index_dict`.
Chunked HomConv execution bounds activation memory without changing the model's
mathematical output.

---

## WordNet Link-Prediction Data Augmentation

The joint WordNet experiment uses one shared DHN + DistMult model, optimizer,
and checkpoint across three graph variants:

```text
no_changes
all_inverse_edges
universal_edges
```

`transitive_edges` is deliberately omitted as a standalone training arm.
However, `universal_edges` still contains both inverse and transitive
augmentations by definition. The canonical four-variant preprocessing archive
is retained so relation IDs, leakage filtering, and fixed invariance candidates
stay compatible with the RGCN augmentation workflow.

Build the shared split archive and DHN bundles:

```bash
python -m preprocess.wordnet.augmentation \
  --data-dir \
  ../INV-GNN/src/baselines/CMPNN/data/raw/wordnet_3hops_augmented_full
```

Validate the selected three variants and model-pattern contract:

```bash
python -m experiments.link_prediction.wordnet_augmentation \
  --config configs/wordnet_augmentation.yaml \
  --device cuda:0 \
  --preflight-only
```

Then run the three configured seeds:

```bash
python -m experiments.link_prediction.wordnet_augmentation \
  --config configs/wordnet_augmentation.yaml \
  --device cuda:0 \
  --output-dir results/dhn_augmentation/WORDNET
```

The default supervised-triple batch size is `65536`. Reducing
`--batch-size` lowers decoder and negative-sampling memory but creates more
full-graph encoder passes and optimizer updates per super-epoch. The two-layer
DHN itself retains `mapping_chunk_size: 100000` and activation checkpointing
from the supplied configuration.

Outputs include `seed_summary.csv`, exact resume states, a shared checkpoint,
per-variant filtered-ranking metrics, fixed-candidate scores, pairwise
invariance, peak process RSS, and CUDA allocated/reserved/peak memory. Resume
with the same command plus `--resume`.

---

## IMDb Node Classification

IMDb node classification predicts Action, Comedy, or Drama for 4,180 movie
nodes. IMDb1-IMDb4 share identical node IDs, features, labels, and the
deterministic seed-`1566911444` 70/10/20 split. IMDb* is the boolean union of
the four baseline adjacencies; the supervision and model configuration remain
unchanged.

On a fresh HPC checkout, first package the tracked intermediate artifacts into
the five DHN bundles:

```bash
python -m preprocess.imdb.node_classification \
  --variant v1,v2,v3,v4,universal
```

Verify the complete benchmark matrix without starting training:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --preflight-only
```

Run all five variants over the lab-standard three seeds:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --preflight \
  --out-dir results/v100/imdb_nc_baseline_retrain
```

This is one command, but it is still the baseline protocol: every
`(variant, seed)` gets its own independently initialized model, optimizer, best
checkpoint, and resource measurements. After those independent runs finish for
a seed, the driver also averages the aligned raw test logits from IMDb1-IMDb4.
IMDb* is trained and reported as a baseline but is not included in the default
fusion. Use `--fusion-variants` to select another ensemble or
`--no-output-fusion` to disable fusion.

The baseline CSVs are written at the output root. Output-fusion results are
written under:

```text
results/v100/imdb_nc_baseline_retrain/output_fusion/
  seed_summary.csv
  output_fusion_raw.csv
  output_fusion_summary.csv
  fusion_vs_variant.csv
  output_fusion_manifest.json
  output_fusion_seed<seed>.npz
```

Each variant directory also contains `seed<seed>.pt` (predictions, aligned
logits, and telemetry) and `best_model_seed<seed>.pt`. The fusion memory report
sums checkpoint/model footprint across its constituent models and takes the
maximum runtime peak for sequential constituent execution. It does not claim a
joint fusion-training peak because output fusion has no joint training phase.

For an incremental rerun that reuses only current-format artifacts:

```bash
python -m experiments.node_classification.benchmark_imdb \
  --skip-existing \
  --out-dir results/v100/imdb_nc_baseline_retrain
```

Legacy cached artifacts missing aligned logits, checkpoints, or telemetry are
detected and retrained. Omit `--skip-existing` to deliberately retrain every
baseline from scratch.

The universal result directory is named `IMDb_universal` (rather than using an
asterisk in a filesystem path), while CSV rows retain the paper label `IMDb*`.
Missing bundles fail the sweep by default; `--allow-missing` is available only
for intentional partial runs.

### Joint IMDb data augmentation

The standalone benchmark above trains a separate model per graph. The joint
augmentation runner instead shares one DHN, optimizer, and checkpoint across
IMDb1-IMDb4. It retains the DHN architecture, loss, optimizer, and chunking
settings from `configs/imdb_nc.yaml`, while matching the RGCN augmentation
schedule:

- one super-epoch visits v1, v2, v3, and v4 once in seeded random order;
- every visit performs one full-batch optimizer update;
- checkpoint selection maximizes mean validation Macro-F1 across all variants;
- test metrics and aligned score tables are written separately per variant;
- pairwise score/prediction invariance is measured from the shared checkpoint;
- the latest super-epoch state supports exact `--resume`;
- per-seed JSON and the aggregate `seed_summary.csv` report checkpoint/model
  bytes, process peak RSS, and train/inference CUDA allocated/reserved peaks.

Build the four baseline bundles, then validate their shared feature,
supervision, and pattern contract:

```bash
python -m preprocess.imdb.node_classification --variant v1,v2,v3,v4

python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc.yaml \
  --preflight-only
```

Run the lab-standard three seeds:

```bash
python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc.yaml \
  --variants v1,v2,v3,v4 \
  --output-dir results/dhn_augmentation/IMDB
```

Resume an interrupted run with the identical configuration:

```bash
python -m experiments.node_classification.imdb_augmentation \
  --config configs/imdb_nc.yaml \
  --variants v1,v2,v3,v4 \
  --output-dir results/dhn_augmentation/IMDB \
  --resume
```

Each `seed_<seed>/` directory contains `shared_checkpoint.pt`,
`latest_training_state.pt`, `training_history.csv`, per-variant test scores and
metrics, `pairwise_invariance.csv`, and `summary.json`. The output root contains
`seed_summary.csv` and `all_seed_summaries.json`.

---

## Freebase Node Classification

The Freebase benchmark predicts one of eight declared classes for labeled BOOK
nodes (`type 0`). It follows the lab baseline preprocessing contract:

- the same labeled node IDs and labels for every graph variant;
- a stratified 60/20/20 split with seed `1566911444`;
- forward edges interpreted as an undirected graph, equivalent to adding reverse
  edges in the baseline loader;
- learned 64-dimensional node embeddings, which are the scalable equivalent of
  projecting identity features.

The baseline variants are `unchanged`, `exact_2`, and `exact_3`. Freebase
`exact_2` already has approximately 49.8 billion target-rooted `p3` mappings,
so all Freebase NC variants consistently use the tractable DHN pattern set
`{p1, c2}`. Bundles record this choice in `meta`. Preprocessing defaults to
`unchanged` and `exact_2`; run `exact_3` separately because its 34 GB edge file
may exceed even the guarded `c2` mapping limit.

### Preprocess

```bash
python -m preprocess.freebase.node_classification \
  --variants unchanged exact_2 \
  --raw-root data/raw/dataset_variant_3hops_filter \
  --out-dir data/preprocessed
```

Expected outputs:

```text
data/preprocessed/Freebase_dhn_nc_unchanged.pt
data/preprocessed/Freebase_dhn_nc_exact_2.pt
```

Preflight/materialize `exact_3` separately. The command aborts instead of
sampling or truncating if it exceeds 20 million unique rooted `c2` mappings:

```bash
python -m preprocess.freebase.node_classification \
  --variants exact_3 \
  --raw-root data/raw/dataset_variant_3hops_filter \
  --out-dir data/preprocessed
```

Only mappings rooted at the 7,954 labeled BOOK nodes are stored. For this
one-layer DHN, that produces the same supervised-node outputs as full-graph
enumeration while avoiding mappings rooted at nodes that never enter the loss.

### Train

```bash
python -m experiments.node_classification.benchmark_freebase \
  --config configs/freebase_nc.yaml \
  --out-dir results/v100/freebase_nc_baseline_retrain
```

The benchmark runs seeds `1566911444`, `20241017`, and `20251017`. It writes
independent per-seed checkpoints, aligned logits, prediction artifacts,
resource telemetry, and:

```text
results/v100/freebase_nc_baseline_retrain/freebase_nc_raw.csv
results/v100/freebase_nc_baseline_retrain/freebase_nc_summary.csv
results/v100/freebase_nc_baseline_retrain/output_fusion/output_fusion_raw.csv
results/v100/freebase_nc_baseline_retrain/output_fusion/output_fusion_summary.csv
```

The default Freebase output fusion averages `No Changes` and `Exact 2-Hop`,
matching the reference output-fusion experiment. `Exact 3-Hop`, when available,
remains an independently reported baseline but is excluded from that default
ensemble.

The summary contains accuracy, macro precision/recall, micro-F1, macro-F1,
training time, elapsed time, time to best validation checkpoint, best epoch,
total epochs trained, CPU peak RSS, model/checkpoint sizes, and training and
inference CUDA allocated/reserved peaks.

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
