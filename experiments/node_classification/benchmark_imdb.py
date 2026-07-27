"""Multi-seed, multi-variant benchmark driver for DHN node classification.

For each (variant, seed) it:
    1. Calls experiments.node_classification.train.run_once(...)
    2. Saves the returned per-run dict to <out_dir>/<variant>/seed<seed>.pt
       (so Table 3 / kendall-tau analyses can reuse y_prob without retraining)
    3. Aggregates accuracy / precision / recall / micro-F1 / macro-F1
       (mean ± std across seeds) into a Table 2 CSV.

IMDb1-IMDb4 and the universal IMDb* mapping are benchmarked by default. Missing
bundles are treated as an error unless ``--allow-missing`` is supplied, so an
HPC sweep cannot silently produce an incomplete comparison.

HPC setup:
    python -m preprocess.imdb.node_classification \
      --variant v1,v2,v3,v4,universal
    python -m experiments.node_classification.benchmark_imdb \
      --skip-existing --preflight
"""
import argparse
import csv
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from dhn.augmentation_utils import RESOURCE_METRIC_KEYS, atomic_torch_save
from experiments.node_classification.output_fusion import (
    cached_artifact_missing_fields,
    write_output_fusion_results,
)
from experiments.node_classification.train import load_config, run_once


# --- variant registry ---------------------------------------------------------
# label -> path to the .pt bundle produced by preprocess.imdb.node_classification
DEFAULT_VARIANTS = {
    'IMDb1': 'data/preprocessed/IMDB_dhn_nc.pt',
    'IMDb2': 'data/preprocessed/IMDB_dhn_nc_t.pt',
    'IMDb3': 'data/preprocessed/IMDB_dhn_nc_t_2.pt',
    'IMDb4': 'data/preprocessed/IMDB_dhn_nc_t_3.pt',
    'IMDb*': 'data/preprocessed/IMDB_dhn_nc_universal.pt',
}
VARIANT_DIR_NAMES = {
    'IMDb1': 'IMDb1',
    'IMDb2': 'IMDb2',
    'IMDb3': 'IMDb3',
    'IMDb4': 'IMDb4',
    'IMDb*': 'IMDb_universal',
}

DEFAULT_SEEDS = [1566911444, 20241017, 20251017]
DEFAULT_FUSION_VARIANTS = ['IMDb1', 'IMDb2', 'IMDb3', 'IMDb4']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', type=str, default='configs/imdb_nc.yaml')
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                   help='Seeds to run per variant (default: lab-standard three seeds)')
    p.add_argument('--variants', type=str, nargs='+', default=None,
                   help='Subset of variant labels to run; default = all available')
    p.add_argument('--out-dir', type=str, default='results/v100/imdb_nc_baseline',
                   help='Where to write per-run artifacts and the summary CSV')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip (variant, seed) pairs whose artifact file already exists')
    p.add_argument(
        '--allow-missing',
        action='store_true',
        help='Warn and skip missing bundles instead of failing the sweep.',
    )
    p.add_argument(
        '--preflight',
        action='store_true',
        help='Print the resolved benchmark matrix before training.',
    )
    p.add_argument(
        '--preflight-only',
        action='store_true',
        help='Validate bundle availability, print the matrix, and exit.',
    )
    p.add_argument(
        '--device',
        default=None,
        help='Override config device, e.g. cuda:0 or cpu.',
    )
    p.add_argument(
        '--fusion-variants',
        nargs='+',
        default=None,
        help='Independent baselines to average; default is IMDb1-IMDb4.',
    )
    p.add_argument(
        '--no-output-fusion',
        action='store_true',
        help='Disable post-hoc averaged-logit output fusion.',
    )
    return p.parse_args()


def resolve_variants(requested, allow_missing=False):
    """Resolve requested bundles, failing by default on an incomplete matrix."""
    if requested is None:
        items = list(DEFAULT_VARIANTS.items())
    else:
        unknown = [v for v in requested if v not in DEFAULT_VARIANTS]
        if unknown:
            raise SystemExit(
                f"Unknown variant labels: {unknown}. "
                f"Expected a subset of {list(DEFAULT_VARIANTS)}"
            )
        items = [(v, DEFAULT_VARIANTS[v]) for v in requested]

    out = {}
    missing = []
    for label, path in items:
        if os.path.exists(path):
            out[label] = path
        else:
            missing.append((label, path))
    if missing and not allow_missing:
        details = "\n".join(
            f"  {label}: {path}" for label, path in missing
        )
        raise SystemExit(
            "Missing required IMDb benchmark bundles:\n"
            f"{details}\n"
            "Generate them with:\n"
            "  python -m preprocess.imdb.node_classification "
            "--variant v1,v2,v3,v4,universal"
        )
    for label, path in missing:
        warnings.warn(f"Variant {label}: bundle not found at {path}; skipping")
    return out


def print_preflight(variants, seeds, out_dir, device):
    print("IMDb DHN node-classification benchmark")
    print(f"  device:   {device}")
    print(f"  seeds:    {seeds}")
    print(f"  out dir:  {out_dir}")
    print("  variants:")
    for label, path in variants.items():
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"    {label:6} {path} ({size_mb:.1f} MiB)")


def metrics_from_run(run, average='macro'):
    """Compute Table-2 metrics for a single run dict from run_once."""
    y_true, y_pred = run['y_true'], run['y_pred']
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def aggregate(rows_for_variant):
    """Given a list of per-seed dicts (metrics + timing + best_epoch),
    return mean and std for each numeric column."""
    keys = [
        'accuracy', 'precision_macro', 'recall_macro',
        'micro_f1', 'macro_f1', 'train_time_s', 'elapsed_time_s',
        'time_to_best_s', 'best_epoch', 'epochs_trained',
        *RESOURCE_METRIC_KEYS,
    ]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows_for_variant], dtype=float)
        agg[f'{k}_mean'] = float(vals.mean())
        agg[f'{k}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return agg


def fmt(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device

    variants = resolve_variants(args.variants, allow_missing=args.allow_missing)
    if not variants:
        print("No variants available — preprocess at least one bundle first.")
        sys.exit(1)

    if args.preflight or args.preflight_only:
        print_preflight(
            variants,
            args.seeds,
            args.out_dir,
            config["device"],
        )
    if args.preflight_only:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Variants:  {list(variants.keys())}")
    print(f"Seeds:     {args.seeds}")
    print(f"Out dir:   {args.out_dir}")

    # Per-variant rows accumulate across seeds, then we aggregate at the end.
    per_variant_rows = {label: [] for label in variants}
    runs_by_seed = {seed: {} for seed in args.seeds}

    for label, data_path in variants.items():
        variant_dir = os.path.join(args.out_dir, VARIANT_DIR_NAMES[label])
        os.makedirs(variant_dir, exist_ok=True)

        for seed in args.seeds:
            run_path = os.path.join(variant_dir, f'seed{seed}.pt')
            checkpoint_path = os.path.join(
                variant_dir, f'best_model_seed{seed}.pt'
            )
            tb_logdir = os.path.join(variant_dir, f'tb_seed{seed}')

            use_cached = args.skip_existing and os.path.exists(run_path)
            if use_cached:
                run = torch.load(run_path, weights_only=False)
                stale_fields = cached_artifact_missing_fields(
                    run, Path(checkpoint_path)
                )
                if stale_fields:
                    print(
                        f"[{label} seed={seed}] cached artifact is missing "
                        f"{stale_fields}; retraining"
                    )
                    use_cached = False
                else:
                    print(
                        f"[{label} seed={seed}] cached, loading {run_path}"
                    )
            if not use_cached:
                print(f"\n[{label} seed={seed}] training (data={data_path})")
                cfg = deepcopy(config)
                run = run_once(
                    cfg,
                    seed=seed,
                    data_path=data_path,
                    logdir=tb_logdir,
                    verbose=False,
                    checkpoint_path=checkpoint_path,
                )
                # tensors-to-arrays already done inside run_once; safe to torch.save
                atomic_torch_save(run, Path(run_path))
                print(f"  best_val={run['best_val_acc']:.4f} "
                      f"test@best={run['best_test_acc']:.4f} "
                      f"epoch={run['best_epoch']} "
                      f"train={run['train_time_s']:.1f}s "
                      f"elapsed={run.get('elapsed_time_s', run['train_time_s']):.1f}s")

            metrics = metrics_from_run(run)
            runs_by_seed[seed][label] = run
            per_variant_rows[label].append({
                **metrics,
                'train_time_s': run['train_time_s'],
                'elapsed_time_s': run.get('elapsed_time_s', run['train_time_s']),
                'time_to_best_s': run.get('time_to_best_s', 0.0),
                'best_epoch': run['best_epoch'],
                'epochs_trained': run['epochs_trained'],
                'seed': seed,
                **{
                    key: run.get(key, float("nan"))
                    for key in RESOURCE_METRIC_KEYS
                },
            })

    # --- write Table 2 CSV ---------------------------------------------------
    csv_path = os.path.join(args.out_dir, 'table2_summary.csv')
    fieldnames = [
        'variant', 'n_seeds',
        'accuracy', 'precision_macro', 'recall_macro',
        'micro_f1', 'macro_f1',
        'train_time_s', 'elapsed_time_s', 'time_to_best_s', 'best_epoch',
        'epochs_trained',
        *RESOURCE_METRIC_KEYS,
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in variants:
            rows = per_variant_rows[label]
            agg = aggregate(rows)
            writer.writerow({
                'variant': label,
                'n_seeds': len(rows),
                'accuracy': fmt(agg['accuracy_mean'], agg['accuracy_std']),
                'precision_macro': fmt(agg['precision_macro_mean'], agg['precision_macro_std']),
                'recall_macro': fmt(agg['recall_macro_mean'], agg['recall_macro_std']),
                'micro_f1': fmt(agg['micro_f1_mean'], agg['micro_f1_std']),
                'macro_f1': fmt(agg['macro_f1_mean'], agg['macro_f1_std']),
                'train_time_s': fmt(agg['train_time_s_mean'], agg['train_time_s_std'], decimals=2),
                'elapsed_time_s': fmt(agg['elapsed_time_s_mean'], agg['elapsed_time_s_std'], decimals=2),
                'time_to_best_s': fmt(agg['time_to_best_s_mean'], agg['time_to_best_s_std'], decimals=2),
                'best_epoch': fmt(agg['best_epoch_mean'], agg['best_epoch_std'], decimals=1),
                'epochs_trained': fmt(
                    agg['epochs_trained_mean'],
                    agg['epochs_trained_std'],
                    decimals=1,
                ),
                **{
                    key: fmt(
                        agg[f'{key}_mean'],
                        agg[f'{key}_std'],
                        decimals=0,
                    )
                    for key in RESOURCE_METRIC_KEYS
                },
            })

    # also dump raw per-seed rows for the appendix / sanity checking
    raw_path = os.path.join(args.out_dir, 'table2_raw.csv')
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'variant', 'seed',
            'accuracy', 'precision_macro', 'recall_macro',
            'micro_f1', 'macro_f1',
            'train_time_s', 'elapsed_time_s', 'time_to_best_s', 'best_epoch',
            'epochs_trained',
            *RESOURCE_METRIC_KEYS,
        ])
        for label in variants:
            for r in per_variant_rows[label]:
                writer.writerow([
                    label, r['seed'],
                    f"{r['accuracy']:.4f}",
                    f"{r['precision_macro']:.4f}",
                    f"{r['recall_macro']:.4f}",
                    f"{r['micro_f1']:.4f}",
                    f"{r['macro_f1']:.4f}",
                    f"{r['train_time_s']:.2f}",
                    f"{r['elapsed_time_s']:.2f}",
                    f"{r['time_to_best_s']:.2f}",
                    r['best_epoch'],
                    r['epochs_trained'],
                    *[
                        f"{r[key]:.0f}"
                        for key in RESOURCE_METRIC_KEYS
                    ],
                ])

    print(f"\nWrote {csv_path}")
    print(f"Wrote {raw_path}")
    if not args.no_output_fusion:
        fusion_variants = (
            args.fusion_variants
            if args.fusion_variants is not None
            else [
                variant
                for variant in DEFAULT_FUSION_VARIANTS
                if variant in variants
            ]
        )
        unavailable = [
            variant for variant in fusion_variants if variant not in variants
        ]
        if unavailable:
            raise SystemExit(
                f"Fusion variants were not trained: {unavailable}"
            )
        if len(fusion_variants) < 2:
            raise SystemExit(
                "Output fusion needs at least two trained variants; select "
                "more variants or pass --no-output-fusion."
            )
        fusion_dir = Path(args.out_dir) / "output_fusion"
        write_output_fusion_results(
            fusion_dir,
            fusion_variants,
            runs_by_seed,
        )
        print(f"Wrote output-fusion artifacts under {fusion_dir}")
    print(
        "Per-run labels, predictions, probabilities, aligned logits, telemetry, "
        "and best checkpoints were saved per (variant, seed)."
    )


if __name__ == '__main__':
    main()
