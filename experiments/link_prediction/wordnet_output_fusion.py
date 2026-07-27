"""Independent WordNet DHN baselines plus post-hoc DistMult score fusion.

Each graph variant receives its own model, optimizer, early stopping, best
checkpoint, and telemetry. After all variants for a seed are complete, their
aligned raw DistMult scores are averaged on the fixed tail-candidate matrix.
This is output fusion, not the shared-model augmentation protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dhn.augmentation_utils import (
    RESOURCE_METRIC_KEYS,
    atomic_write_csv,
    torch_load_full,
    write_json,
)
from experiments.link_prediction.wordnet import (
    run_one_seed,
    write_kg_summary_csv,
)
from experiments.node_classification.output_fusion import (
    full_logit_kendall_tau,
    output_fusion_memory,
)


DEFAULT_VARIANTS = (
    "no_changes",
    "all_inverse_edges",
    "universal_edges",
)
KNOWN_VARIANTS = (
    "no_changes",
    "all_inverse_edges",
    "transitive_edges",
    "universal_edges",
)
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)


def bundle_path(bundle_dir: Path, variant: str) -> Path:
    return bundle_dir / f"WordNet3Hop_dhn_lp_{variant}.pt"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_bundles(
    variants: Sequence[str], bundle_dir: Path
) -> dict[str, dict[str, Any]]:
    """Load bundles on CPU and prove the score-fusion alignment contract."""
    unknown = sorted(set(variants) - set(KNOWN_VARIANTS))
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}; expected a subset of {KNOWN_VARIANTS}"
        )
    if len(variants) != len(set(variants)):
        raise ValueError("Variants must not contain duplicates")
    if len(variants) < 2:
        raise ValueError("WordNet output fusion requires at least two variants")

    bundles: dict[str, dict[str, Any]] = {}
    for variant in variants:
        path = bundle_path(bundle_dir, variant)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Build the four compatible bundles with "
                "python -m preprocess.wordnet.augmentation."
            )
        bundle = torch_load_full(path, map_location="cpu")
        bundle["_path"] = str(path)
        bundles[variant] = bundle

    reference = bundles[variants[0]]
    reference_meta = reference["meta"]
    reference_val = _as_numpy(reference["splits"]["val"])
    reference_test = _as_numpy(reference["splits"]["test"])
    reference_vocab = reference.get("vocab")
    for variant in variants[1:]:
        current = bundles[variant]
        current_meta = current["meta"]
        for field in ("num_entities", "num_relations"):
            if current_meta[field] != reference_meta[field]:
                raise ValueError(
                    f"{field} differs for {variant}: "
                    f"{current_meta[field]} != {reference_meta[field]}"
                )
        if not np.array_equal(
            reference_val, _as_numpy(current["splits"]["val"])
        ):
            raise ValueError(f"Validation triples differ for {variant}")
        if not np.array_equal(
            reference_test, _as_numpy(current["splits"]["test"])
        ):
            raise ValueError(f"Test triples differ for {variant}")
        if reference_vocab is not None and current.get("vocab") != reference_vocab:
            raise ValueError(f"Entity/relation vocabulary differs for {variant}")
    return bundles


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(logits)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1.0 / (
        1.0 + np.exp(-logits[nonnegative])
    )
    exp_values = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = exp_values / (1.0 + exp_values)
    return probabilities


def candidate_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    hits_k: Sequence[int],
) -> dict[str, float]:
    """Metrics on the explicitly labeled fixed tail-candidate matrix."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.shape != labels.shape or logits.ndim != 2:
        raise ValueError(
            f"Expected aligned 2-D scores/labels, got "
            f"{logits.shape} and {labels.shape}"
        )
    positives_per_query = labels.sum(axis=1)
    if not np.all(positives_per_query == 1):
        raise ValueError("Each candidate query must contain exactly one positive")

    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    probabilities = stable_sigmoid(flat_logits)
    predictions = (flat_logits > 0).astype(np.int64)
    true_indices = labels.argmax(axis=1)
    true_scores = logits[np.arange(len(logits)), true_indices]
    ranks = 1 + (logits > true_scores[:, None]).sum(axis=1)

    metrics = {
        "candidate_BCE": float(
            np.mean(np.logaddexp(0.0, flat_logits) - flat_labels * flat_logits)
        ),
        "candidate_AUC": float(roc_auc_score(flat_labels, flat_logits)),
        "candidate_AP": float(
            average_precision_score(flat_labels, flat_logits)
        ),
        "candidate_Accuracy": float(
            accuracy_score(flat_labels, predictions)
        ),
        "candidate_Precision_macro": float(
            precision_score(
                flat_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "candidate_Recall_macro": float(
            recall_score(
                flat_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "candidate_Micro_F1": float(
            f1_score(
                flat_labels,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
        "candidate_Macro_F1": float(
            f1_score(
                flat_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "candidate_MRR_tail": float(np.mean(1.0 / ranks)),
    }
    for k in hits_k:
        metrics[f"candidate_Hits@{k}_tail"] = float(np.mean(ranks <= k))
    return metrics


def load_score_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def fuse_seed(
    seed: int,
    variants: Sequence[str],
    score_artifacts: Mapping[str, Mapping[str, np.ndarray]],
    runs_by_variant: Mapping[str, Mapping[str, Any]],
    hits_k: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    missing = [
        variant
        for variant in variants
        if variant not in score_artifacts or variant not in runs_by_variant
    ]
    if missing:
        raise ValueError(f"seed={seed} lacks variants {missing}")

    reference = score_artifacts[variants[0]]
    required_arrays = {
        "test_pos",
        "candidate_ids",
        "scores",
        "labels",
        "candidate_rng_seed",
    }
    missing_arrays = sorted(required_arrays - set(reference))
    if missing_arrays:
        raise ValueError(
            f"{variants[0]}/seed={seed} lacks arrays {missing_arrays}"
        )

    test_pos = np.asarray(reference["test_pos"], dtype=np.int64)
    candidate_ids = np.asarray(reference["candidate_ids"], dtype=np.int64)
    labels = np.asarray(reference["labels"], dtype=np.int64)
    candidate_seed = np.asarray(reference["candidate_rng_seed"])
    scores_by_variant: dict[str, np.ndarray] = {}
    for variant in variants:
        artifact = score_artifacts[variant]
        for field, expected in (
            ("test_pos", test_pos),
            ("candidate_ids", candidate_ids),
            ("labels", labels),
            ("candidate_rng_seed", candidate_seed),
        ):
            if field not in artifact or not np.array_equal(
                np.asarray(artifact[field]), expected
            ):
                raise ValueError(
                    f"{field} is not aligned for {variant}/seed={seed}"
                )
        scores = np.asarray(artifact["scores"], dtype=np.float32)
        if scores.shape != labels.shape:
            raise ValueError(
                f"Score shape differs for {variant}/seed={seed}: "
                f"{scores.shape} != {labels.shape}"
            )
        scores_by_variant[variant] = scores

    averaged_scores = np.mean(
        np.stack([scores_by_variant[variant] for variant in variants]),
        axis=0,
    )
    comparisons = []
    fused_predictions = averaged_scores > 0
    for variant in variants:
        scores = scores_by_variant[variant]
        comparisons.append(
            {
                "seed": int(seed),
                "variant": variant,
                "FullKendallTau_fusion_vs_variant": full_logit_kendall_tau(
                    averaged_scores, scores
                ),
                "prediction_agreement_fusion_vs_variant": float(
                    np.mean(fused_predictions == (scores > 0))
                ),
            }
        )
    finite_taus = [
        row["FullKendallTau_fusion_vs_variant"]
        for row in comparisons
        if np.isfinite(row["FullKendallTau_fusion_vs_variant"])
    ]
    result = {
        "seed": int(seed),
        "variants": ",".join(variants),
        "candidate_queries": int(labels.shape[0]),
        "candidates_per_query": int(labels.shape[1]),
        **candidate_metrics(averaged_scores, labels, hits_k),
        "FullKendallTau": (
            float(np.mean(finite_taus)) if finite_taus else float("nan")
        ),
        **output_fusion_memory(
            {variant: runs_by_variant[variant] for variant in variants}
        ),
    }
    arrays = {
        "test_pos": test_pos,
        "candidate_ids": candidate_ids,
        "labels": labels,
        "averaged_scores": averaged_scores,
        "candidate_rng_seed": candidate_seed,
        "variant_names": np.asarray(variants),
    }
    for index, variant in enumerate(variants):
        arrays[f"variant_{index}_scores"] = scores_by_variant[variant]
    return result, comparisons, arrays


def write_fusion_results(
    output_dir: Path,
    variants: Sequence[str],
    seeds: Sequence[int],
    score_paths_by_seed: Mapping[int, Mapping[str, Path]],
    runs_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    hits_k: Sequence[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    comparisons = []
    for seed in seeds:
        artifacts = {
            variant: load_score_artifact(score_paths_by_seed[seed][variant])
            for variant in variants
        }
        row, seed_comparisons, arrays = fuse_seed(
            seed, variants, artifacts, runs_by_seed[seed], hits_k
        )
        rows.append(row)
        comparisons.extend(seed_comparisons)
        np.savez_compressed(
            output_dir / f"output_fusion_seed{seed}.npz", **arrays
        )

    seed_frame = pd.DataFrame(rows)
    atomic_write_csv(seed_frame, output_dir / "seed_summary.csv")
    atomic_write_csv(seed_frame, output_dir / "output_fusion_raw.csv")
    atomic_write_csv(
        pd.DataFrame(comparisons), output_dir / "fusion_vs_variant.csv"
    )
    summary_rows = []
    for column in seed_frame.columns:
        if column in {"seed", "variants"}:
            continue
        values = seed_frame[column].to_numpy(dtype=np.float64)
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
            "protocol": (
                "posthoc_average_of_independent_wordnet_distmult_scores"
            ),
            "variants": list(variants),
            "seeds": [int(seed) for seed in seeds],
            "score_domain": "raw_pre_sigmoid_distmult_scores",
            "evaluation_scope": (
                "shared_fixed_tail_candidates_only; constituent baseline "
                "CSVs separately report full filtered all-entity metrics"
            ),
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
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/wordnet_lp.yaml"))
    parser.add_argument(
        "--bundle-dir", type=Path, default=Path("data/preprocessed")
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v100/wordnet_lp_baseline_retrain"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not config.get("artifacts", {}).get("save_best_checkpoint", False):
        raise SystemExit("WordNet output fusion requires save_best_checkpoint=true")
    if not config.get("artifacts", {}).get("kendall", {}).get("enabled", False):
        raise SystemExit("WordNet output fusion requires artifacts.kendall.enabled=true")

    variants = list(args.variants)
    bundles = validate_bundles(variants, args.bundle_dir)
    seeds = (
        list(args.seeds)
        if args.seeds is not None
        else [int(seed) for seed in config.get("seeds", DEFAULT_SEEDS)]
    )
    device = args.device or config.get("device", "cuda:0")
    print("WordNet independent-baseline output fusion")
    print(f"  variants: {variants}")
    print(f"  seeds:    {seeds}")
    print(f"  device:   {device}")
    print(f"  output:   {args.output_dir}")
    for variant in variants:
        path = Path(bundles[variant]["_path"])
        print(f"  {variant}: {path} ({path.stat().st_size / 2**20:.1f} MiB)")
    if args.preflight_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_by_seed: dict[int, dict[str, Mapping[str, Any]]] = {
        seed: {} for seed in seeds
    }
    score_paths_by_seed: dict[int, dict[str, Path]] = {
        seed: {} for seed in seeds
    }
    hits_k = tuple(int(k) for k in config["eval"]["hits_k"])

    for variant in variants:
        variant_dir = args.output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        bundle = bundles[variant]
        bundle_file = Path(bundle["_path"])
        dataset_slug = bundle["meta"]["dataset_slug"]
        per_seed = []
        for seed in seeds:
            metrics_path = variant_dir / f"metrics_seed{seed}.json"
            score_path = (
                variant_dir
                / f"kendall_tail_scores_{dataset_slug}_seed{seed}.npz"
            )
            checkpoint_path = (
                variant_dir
                / "checkpoints"
                / f"best_model_{dataset_slug}_seed{seed}.pt"
            )
            use_cached = (
                args.skip_existing
                and metrics_path.is_file()
                and score_path.is_file()
                and checkpoint_path.is_file()
            )
            if use_cached:
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                missing_resources = [
                    field
                    for field in RESOURCE_METRIC_KEYS
                    if field not in metrics
                ]
                if missing_resources:
                    print(
                        f"[{variant} seed={seed}] cache lacks "
                        f"{missing_resources}; retraining"
                    )
                    use_cached = False
                else:
                    print(f"[{variant} seed={seed}] using cached artifacts")
            if not use_cached:
                print(f"\n[{variant} seed={seed}] independent training")
                metrics = run_one_seed(
                    config,
                    str(bundle_file),
                    seed,
                    device,
                    str(variant_dir),
                    verbose=True,
                )
                write_json(metrics_path, metrics)
            per_seed.append(metrics)
            runs_by_seed[seed][variant] = metrics
            score_paths_by_seed[seed][variant] = score_path

        write_kg_summary_csv(
            str(variant_dir / f"lp_summary_{dataset_slug}.csv"),
            per_seed,
            hits_k,
        )

    fusion_dir = args.output_dir / "output_fusion"
    write_fusion_results(
        fusion_dir,
        variants,
        seeds,
        score_paths_by_seed,
        runs_by_seed,
        hits_k,
    )
    print(f"Wrote independent baselines under {args.output_dir}")
    print(f"Wrote output fusion under {fusion_dir}")


if __name__ == "__main__":
    main()
