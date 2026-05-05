"""Multi-seed, multi-variant benchmark driver for DHN node classification.

For each (variant, seed) it:
    1. Calls train_nc.run_once(...)
    2. Saves the returned per-run dict to <out_dir>/<variant>/seed<seed>.pt
       (so Table 3 / kendall-tau analyses can reuse y_prob without retraining)
    3. Aggregates accuracy / precision / recall / micro-F1 / macro-F1
       (mean ± std across seeds) into a Table 2 CSV.

Variants whose data bundle does not yet exist are skipped with a warning,
so this can be re-run incrementally as new IMDb variants get preprocessed.
"""
import argparse
import csv
import os
import sys
import warnings
from copy import deepcopy

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_nc import load_config, run_once


# --- variant registry ---------------------------------------------------------
# label -> path to the .pt bundle produced by preprocess/preprocess_IMDB_for_dhn_nc.py
DEFAULT_VARIANTS = {
    'IMDb1': 'data/preprocessed/IMDB_dhn_nc.pt',
    'IMDb2': 'data/preprocessed/IMDB_dhn_nc_t.pt',
    'IMDb3': 'data/preprocessed/IMDB_dhn_nc_t_2.pt',
    'IMDb4': 'data/preprocessed/IMDB_dhn_nc_t_3.pt',
}

DEFAULT_SEEDS = [42, 43, 44]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', type=str, default='configs/imdb_nc.yaml')
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                   help='Seeds to run per variant (default: 42 43 44)')
    p.add_argument('--variants', type=str, nargs='+', default=None,
                   help='Subset of variant labels to run; default = all available')
    p.add_argument('--out-dir', type=str, default='runs/benchmark_imdb_nc',
                   help='Where to write per-run artifacts and the summary CSV')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip (variant, seed) pairs whose artifact file already exists')
    return p.parse_args()


def resolve_variants(requested):
    """Return {label: path} for variants whose bundle file exists. Warn for missing."""
    if requested is None:
        items = list(DEFAULT_VARIANTS.items())
    else:
        items = [(v, DEFAULT_VARIANTS[v]) for v in requested if v in DEFAULT_VARIANTS]
        unknown = [v for v in requested if v not in DEFAULT_VARIANTS]
        if unknown:
            warnings.warn(f"Unknown variant labels (skipped): {unknown}")
    out = {}
    for label, path in items:
        if os.path.exists(path):
            out[label] = path
        else:
            warnings.warn(f"Variant {label}: bundle not found at {path}; skipping")
    return out


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
    """Given a list of per-seed dicts (metrics + train_time_s + best_epoch),
    return mean and std for each numeric column."""
    keys = [
        'accuracy', 'precision_macro', 'recall_macro',
        'micro_f1', 'macro_f1', 'train_time_s', 'best_epoch',
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

    variants = resolve_variants(args.variants)
    if not variants:
        print("No variants available — preprocess at least one bundle first.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Variants:  {list(variants.keys())}")
    print(f"Seeds:     {args.seeds}")
    print(f"Out dir:   {args.out_dir}")

    # Per-variant rows accumulate across seeds, then we aggregate at the end.
    per_variant_rows = {label: [] for label in variants}

    for label, data_path in variants.items():
        variant_dir = os.path.join(args.out_dir, label)
        os.makedirs(variant_dir, exist_ok=True)

        for seed in args.seeds:
            run_path = os.path.join(variant_dir, f'seed{seed}.pt')
            tb_logdir = os.path.join(variant_dir, f'tb_seed{seed}')

            if args.skip_existing and os.path.exists(run_path):
                print(f"[{label} seed={seed}] cached, loading {run_path}")
                run = torch.load(run_path)
            else:
                print(f"\n[{label} seed={seed}] training (data={data_path})")
                cfg = deepcopy(config)
                run = run_once(
                    cfg,
                    seed=seed,
                    data_path=data_path,
                    logdir=tb_logdir,
                    verbose=False,
                )
                # tensors-to-arrays already done inside run_once; safe to torch.save
                torch.save(run, run_path)
                print(f"  best_val={run['best_val_acc']:.4f} "
                      f"test@best={run['best_test_acc']:.4f} "
                      f"epoch={run['best_epoch']} "
                      f"time={run['train_time_s']:.1f}s")

            metrics = metrics_from_run(run)
            per_variant_rows[label].append({
                **metrics,
                'train_time_s': run['train_time_s'],
                'best_epoch': run['best_epoch'],
                'seed': seed,
            })

    # --- write Table 2 CSV ---------------------------------------------------
    csv_path = os.path.join(args.out_dir, 'table2_summary.csv')
    fieldnames = [
        'variant', 'n_seeds',
        'accuracy', 'precision_macro', 'recall_macro',
        'micro_f1', 'macro_f1',
        'train_time_s', 'best_epoch',
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
                'best_epoch': fmt(agg['best_epoch_mean'], agg['best_epoch_std'], decimals=1),
            })

    # also dump raw per-seed rows for the appendix / sanity checking
    raw_path = os.path.join(args.out_dir, 'table2_raw.csv')
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'variant', 'seed',
            'accuracy', 'precision_macro', 'recall_macro',
            'micro_f1', 'macro_f1', 'train_time_s', 'best_epoch',
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
                    r['best_epoch'],
                ])

    print(f"\nWrote {csv_path}")
    print(f"Wrote {raw_path}")
    print("Per-run artifacts (y_true / y_pred / y_prob) saved per (variant, seed); "
          "reuse them for Table 3 (kendall-tau) once the definition is settled.")


if __name__ == '__main__':
    main()
