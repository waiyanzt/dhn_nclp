"""Run multi-seed DHN node classification on Freebase graph variants."""
from __future__ import annotations

import argparse
import csv
import os
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dhn.augmentation_utils import RESOURCE_METRIC_KEYS, atomic_torch_save
from experiments.node_classification.output_fusion import (
    cached_artifact_missing_fields,
    write_output_fusion_results,
)
from experiments.node_classification.train import load_config, run_once


DEFAULT_VARIANTS = {
    "No Changes": "data/preprocessed/Freebase_dhn_nc_unchanged.pt",
    "Exact 2-Hop": "data/preprocessed/Freebase_dhn_nc_exact_2.pt",
    "Exact 3-Hop": "data/preprocessed/Freebase_dhn_nc_exact_3.pt",
}
DEFAULT_SEEDS = [1566911444, 20241017, 20251017]
DEFAULT_FUSION_VARIANTS = ["No Changes", "Exact 2-Hop"]
METRIC_KEYS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "micro_f1",
    "macro_f1",
    "train_time_s",
    "elapsed_time_s",
    "time_to_best_s",
    "best_epoch",
    "epochs_trained",
    *RESOURCE_METRIC_KEYS,
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/freebase_nc.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--out-dir", default="results/v100/freebase_nc_baseline"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate selected bundle availability, print the run matrix, and exit.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--fusion-variants",
        nargs="+",
        default=None,
        help="Independent baselines to average; default: No Changes + Exact 2-Hop",
    )
    parser.add_argument(
        "--no-output-fusion",
        action="store_true",
        help="Disable post-hoc averaged-logit output fusion",
    )
    return parser.parse_args()


def resolve_variants(requested):
    names = list(DEFAULT_VARIANTS) if requested is None else requested
    unknown = [name for name in names if name not in DEFAULT_VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown Freebase variants: {unknown}")
    resolved = {}
    for name in names:
        path = DEFAULT_VARIANTS[name]
        if os.path.isfile(path):
            resolved[name] = path
        else:
            warnings.warn(f"Missing bundle for {name}: {path}")
    return resolved


def metrics_from_run(run):
    y_true, y_pred = run["y_true"], run["y_pred"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def aggregate(rows):
    result = {}
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = (
            float(values.mean()),
            float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        )
    return result


def formatted(mean, std, decimals):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    variants = resolve_variants(args.variants)
    if not variants:
        raise SystemExit("No Freebase bundles are available")

    if args.preflight_only:
        print("Freebase DHN node-classification baseline preflight")
        print(f"  device:   {config['device']}")
        print(f"  seeds:    {args.seeds}")
        print(f"  output:   {args.out_dir}")
        for variant, bundle_path in variants.items():
            size_mib = Path(bundle_path).stat().st_size / 2**20
            print(f"  {variant}: {bundle_path} ({size_mib:.1f} MiB)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = {}
    runs_by_seed = {seed: {} for seed in args.seeds}
    for variant, bundle_path in variants.items():
        variant_dir = os.path.join(
            args.out_dir, variant.lower().replace(" ", "_").replace("-", "")
        )
        os.makedirs(variant_dir, exist_ok=True)
        rows = []

        for seed in args.seeds:
            artifact_path = os.path.join(variant_dir, f"seed{seed}.pt")
            checkpoint_path = os.path.join(
                variant_dir, f"best_model_seed{seed}.pt"
            )
            use_cached = args.skip_existing and os.path.isfile(artifact_path)
            if use_cached:
                run = torch.load(artifact_path, weights_only=False)
                stale_fields = cached_artifact_missing_fields(
                    run, Path(checkpoint_path)
                )
                if stale_fields:
                    print(
                        f"[{variant} seed={seed}] cached artifact is missing "
                        f"{stale_fields}; retraining"
                    )
                    use_cached = False
                else:
                    print(
                        f"[{variant} seed={seed}] loaded {artifact_path}"
                    )
            if not use_cached:
                run = run_once(
                    deepcopy(config),
                    seed=seed,
                    data_path=bundle_path,
                    logdir=os.path.join(variant_dir, f"tb_seed{seed}"),
                    verbose=False,
                    checkpoint_path=checkpoint_path,
                )
                atomic_torch_save(run, Path(artifact_path))
                print(
                    f"[{variant} seed={seed}] "
                    f"best_val={run['best_val_acc']:.4f} "
                    f"test={run['best_test_acc']:.4f} "
                    f"train={run['train_time_s']:.2f}s"
                )

            runs_by_seed[seed][variant] = run
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **metrics_from_run(run),
                    "train_time_s": run["train_time_s"],
                    "elapsed_time_s": run["elapsed_time_s"],
                    "time_to_best_s": run["time_to_best_s"],
                    "best_epoch": run["best_epoch"],
                    "epochs_trained": run["epochs_trained"],
                    **{
                        key: run.get(key, float("nan"))
                        for key in RESOURCE_METRIC_KEYS
                    },
                }
            )
        all_rows[variant] = rows

    raw_path = os.path.join(args.out_dir, "freebase_nc_raw.csv")
    raw_fields = ["variant", "seed", *METRIC_KEYS]
    with open(raw_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(row for rows in all_rows.values() for row in rows)

    summary_path = os.path.join(args.out_dir, "freebase_nc_summary.csv")
    summary_fields = ["variant", "n_seeds", *METRIC_KEYS]
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for variant, rows in all_rows.items():
            stats = aggregate(rows)
            output = {"variant": variant, "n_seeds": len(rows)}
            for key, (mean, std) in stats.items():
                decimals = (
                    0
                    if key in RESOURCE_METRIC_KEYS
                    else 2
                    if key.endswith("_s")
                    else 1
                    if "epoch" in key
                    else 4
                )
                output[key] = formatted(mean, std, decimals)
            writer.writerow(output)

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
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


if __name__ == "__main__":
    main()
