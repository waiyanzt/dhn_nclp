"""Post-hoc logit averaging for independently trained NC baselines.

This module never trains across graph variants. It consumes aligned test logits
from one independently trained model per variant and seed, averages those
logits, and writes ensemble metrics and auditable per-seed artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from dhn.augmentation_utils import (
    RESOURCE_METRIC_KEYS,
    atomic_write_csv,
    write_json,
)


ADDITIVE_FOOTPRINT_FIELDS = (
    "parameter_bytes",
    "buffer_bytes",
    "static_model_bytes",
    "checkpoint_bytes",
)
TRAINING_GPU_FIELDS = (
    "training_gpu_allocated_bytes",
    "training_gpu_reserved_bytes",
    "training_gpu_peak_allocated_bytes",
    "training_gpu_peak_reserved_bytes",
)
INFERENCE_GPU_FIELDS = (
    "inference_gpu_allocated_bytes",
    "inference_gpu_reserved_bytes",
    "inference_gpu_peak_allocated_bytes",
    "inference_gpu_peak_reserved_bytes",
)


def cached_artifact_missing_fields(
    run: Mapping[str, Any], checkpoint_path: Path
) -> list[str]:
    """Identify legacy artifacts that cannot support the new baseline report."""
    required = {"y_logits", "test_node_ids", *RESOURCE_METRIC_KEYS}
    missing = [
        field
        for field in sorted(required)
        if field not in run or run[field] is None
    ]
    if not Path(checkpoint_path).is_file():
        missing.append("checkpoint_file")
    return missing


def _safe_tau(left: np.ndarray, right: np.ndarray) -> float:
    statistic = kendalltau(
        np.asarray(left),
        np.asarray(right),
        variant="b",
        nan_policy="omit",
    ).statistic
    return (
        float(statistic)
        if statistic is not None and np.isfinite(statistic)
        else float("nan")
    )


def full_logit_kendall_tau(
    logits_a: np.ndarray, logits_b: np.ndarray
) -> float:
    """Mean per-node Kendall tau-b over the complete class ranking."""
    left = np.asarray(logits_a)
    right = np.asarray(logits_b)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"Expected aligned 2-D logits, got {left.shape} and {right.shape}"
        )
    values = [
        _safe_tau(row_a, row_b) for row_a, row_b in zip(left, right)
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision_macro": float(
            precision_score(
                labels, predictions, average="macro", zero_division=0
            )
        ),
        "Recall_macro": float(
            recall_score(
                labels, predictions, average="macro", zero_division=0
            )
        ),
        "Micro_F1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "Macro_F1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def output_fusion_memory(
    runs_by_variant: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    """Report storage sums and sequential-execution peaks without conflation."""
    runs = list(runs_by_variant.values())
    required = {
        *ADDITIVE_FOOTPRINT_FIELDS,
        "process_peak_rss_bytes",
        *TRAINING_GPU_FIELDS,
        *INFERENCE_GPU_FIELDS,
    }
    for variant, run in runs_by_variant.items():
        missing = sorted(required - set(run))
        if missing:
            raise RuntimeError(
                f"{variant} lacks resource fields {missing}; rerun that "
                "baseline with telemetry enabled."
            )
    output: dict[str, float] = {}
    for field in ADDITIVE_FOOTPRINT_FIELDS:
        output[f"fusion_{field}_sum"] = float(
            sum(float(run[field]) for run in runs)
        )
    output["constituent_process_peak_rss_bytes_max"] = float(
        max(float(run["process_peak_rss_bytes"]) for run in runs)
    )
    for field in TRAINING_GPU_FIELDS:
        # Output fusion has no joint training phase. This describes the largest
        # independently trained constituent only.
        output[f"constituent_{field}_max"] = float(
            max(float(run[field]) for run in runs)
        )
    for field in INFERENCE_GPU_FIELDS:
        # Valid when constituent models are loaded/executed sequentially.
        output[f"sequential_{field}_max"] = float(
            max(float(run[field]) for run in runs)
        )
    return output


def fuse_seed(
    seed: int,
    variants: Sequence[str],
    runs_by_variant: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    missing = [variant for variant in variants if variant not in runs_by_variant]
    if missing:
        raise ValueError(f"seed={seed} is missing fusion variants: {missing}")

    reference = runs_by_variant[variants[0]]
    if reference.get("y_logits") is None or reference.get("test_node_ids") is None:
        raise RuntimeError(
            "Cached baseline artifacts predate aligned-logit recording. "
            "Rerun without --skip-existing."
        )
    labels = np.asarray(reference["y_true"], dtype=np.int64)
    node_ids = np.asarray(reference["test_node_ids"], dtype=np.int64)
    logits_by_variant: dict[str, np.ndarray] = {}
    for variant in variants:
        run = runs_by_variant[variant]
        if run.get("y_logits") is None or run.get("test_node_ids") is None:
            raise RuntimeError(
                f"{variant}/seed={seed} lacks y_logits or test_node_ids; "
                "rerun that baseline."
            )
        current_labels = np.asarray(run["y_true"], dtype=np.int64)
        current_ids = np.asarray(run["test_node_ids"], dtype=np.int64)
        logits = np.asarray(run["y_logits"], dtype=np.float32)
        if not np.array_equal(labels, current_labels):
            raise ValueError(f"Test labels differ for {variant}/seed={seed}")
        if not np.array_equal(node_ids, current_ids):
            raise ValueError(f"Test node IDs differ for {variant}/seed={seed}")
        if logits.ndim != 2 or logits.shape[0] != len(labels):
            raise ValueError(f"Invalid logit shape for {variant}/seed={seed}")
        logits_by_variant[variant] = logits

    shapes = {logits.shape for logits in logits_by_variant.values()}
    if len(shapes) != 1:
        raise ValueError(f"Logit shapes differ for seed={seed}: {shapes}")
    averaged_logits = np.mean(
        np.stack([logits_by_variant[variant] for variant in variants]),
        axis=0,
    )
    predictions = averaged_logits.argmax(axis=1)
    comparisons = []
    for variant in variants:
        constituent = logits_by_variant[variant]
        comparisons.append(
            {
                "seed": int(seed),
                "variant": variant,
                "kendall_tau_fusion_vs_variant": full_logit_kendall_tau(
                    averaged_logits, constituent
                ),
                "prediction_agreement_fusion_vs_variant": float(
                    np.mean(
                        predictions == constituent.argmax(axis=1)
                    )
                ),
            }
        )
    finite_taus = [
        row["kendall_tau_fusion_vs_variant"]
        for row in comparisons
        if np.isfinite(row["kendall_tau_fusion_vs_variant"])
    ]
    fused_runs = {
        variant: runs_by_variant[variant] for variant in variants
    }
    result = {
        "seed": int(seed),
        "variants": ",".join(variants),
        **classification_metrics(labels, predictions),
        "FullKendallTau": (
            float(np.mean(finite_taus)) if finite_taus else float("nan")
        ),
        **output_fusion_memory(fused_runs),
    }
    arrays = {
        "test_node_ids": node_ids,
        "labels": labels,
        "averaged_logits": averaged_logits,
        "predictions": predictions,
        "variant_names": np.asarray(variants),
    }
    for index, variant in enumerate(variants):
        arrays[f"variant_{index}_logits"] = logits_by_variant[variant]
    return result, comparisons, arrays


def write_output_fusion_results(
    output_dir: Path,
    variants: Sequence[str],
    runs_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if len(variants) < 2:
        raise ValueError("Output fusion requires at least two variants")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for seed in sorted(runs_by_seed):
        row, seed_comparisons, arrays = fuse_seed(
            seed, variants, runs_by_seed[seed]
        )
        rows.append(row)
        comparisons.extend(seed_comparisons)
        np.savez_compressed(
            output_dir / f"output_fusion_seed{seed}.npz",
            **arrays,
        )

    raw_frame = pd.DataFrame(rows)
    atomic_write_csv(raw_frame, output_dir / "output_fusion_raw.csv")
    # Keep the familiar augmentation-runner filename as a wide, one-row-per-seed
    # entry point while retaining the more explicit fusion-specific name.
    atomic_write_csv(raw_frame, output_dir / "seed_summary.csv")
    atomic_write_csv(
        pd.DataFrame(comparisons),
        output_dir / "fusion_vs_variant.csv",
    )
    numeric_columns = [
        column
        for column in raw_frame.columns
        if column not in {"seed", "variants"}
    ]
    summary_rows = []
    for column in numeric_columns:
        values = raw_frame[column].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "metric": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "n_seeds": int(len(values)),
            }
        )
    atomic_write_csv(
        pd.DataFrame(summary_rows),
        output_dir / "output_fusion_summary.csv",
    )
    write_json(
        output_dir / "output_fusion_manifest.json",
        {
            "protocol": "posthoc_average_of_independent_baseline_logits",
            "variants": list(variants),
            "seeds": sorted(int(seed) for seed in runs_by_seed),
            "memory_semantics": {
                "footprint_fields": "sum across constituent checkpoints/models",
                "inference_fields": (
                    "max across constituents under sequential execution"
                ),
                "training_fields": (
                    "max independent constituent; output fusion has no joint "
                    "training phase"
                ),
            },
            "artifacts": {
                "seed_summary": "seed_summary.csv",
                "per_seed_metrics": "output_fusion_raw.csv",
                "aggregate_metrics": "output_fusion_summary.csv",
                "comparisons": "fusion_vs_variant.csv",
                "aligned_arrays": "output_fusion_seed<seed>.npz",
            },
        },
    )
    return rows
