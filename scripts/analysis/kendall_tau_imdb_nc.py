"""Pairwise test-set Kendall tau-b for saved DHN IMDb NC artifacts.

For each matched seed and test node, compare the ranking of the three class
scores between two graph variants. Average over test nodes, then report the
mean and sample standard deviation over seeds.

Usage:
    python -m scripts.analysis.kendall_tau_imdb_nc
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau


DEFAULT_VARIANTS = ("IMDb1", "IMDb2", "IMDb3", "IMDb4")
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)


def load_artifact(root: Path, variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / variant / f"seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = torch.load(path, weights_only=False, map_location="cpu")
    return np.asarray(artifact["y_prob"]), np.asarray(artifact["y_true"])


def mean_node_tau(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if scores_a.shape != scores_b.shape:
        raise ValueError(f"Score shapes differ: {scores_a.shape} vs {scores_b.shape}")
    taus = np.asarray([
        kendalltau(a, b, variant="b", nan_policy="omit").statistic
        for a, b in zip(scores_a, scores_b)
    ], dtype=np.float64)
    return float(np.nanmean(taus))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default="results/v100/imdb_nc_baseline",
        help="Directory containing <variant>/seed<seed>.pt artifacts.",
    )
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        default="reports/kendall_tau_imdb_nc_test.csv",
        help="Summary CSV path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    loaded = {
        (variant, seed): load_artifact(root, variant, seed)
        for variant in args.variants
        for seed in args.seeds
    }

    rows = []
    for variant_a, variant_b in itertools.combinations(args.variants, 2):
        seed_taus = []
        for seed in args.seeds:
            scores_a, labels_a = loaded[(variant_a, seed)]
            scores_b, labels_b = loaded[(variant_b, seed)]
            if not np.array_equal(labels_a, labels_b):
                raise ValueError(
                    f"Test labels/order differ for {variant_a}, {variant_b}, seed {seed}"
                )
            tau = mean_node_tau(scores_a, scores_b)
            seed_taus.append(tau)
            print(f"{variant_a} vs {variant_b} seed={seed}: test tau={tau:.6f}")

        values = np.asarray(seed_taus, dtype=np.float64)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append({
            "variant_a": variant_a,
            "variant_b": variant_b,
            "n_seeds": len(values),
            "kendall_tau_test_mean": mean,
            "kendall_tau_test_std": std,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary (mean +/- sample std over matched seeds)")
    for row in rows:
        print(
            f"{row['variant_a']} vs {row['variant_b']}: "
            f"{row['kendall_tau_test_mean']:.4f} +/- "
            f"{row['kendall_tau_test_std']:.4f}"
        )

    print("\nLaTeX rows")
    for row in rows:
        print(
            f"{row['variant_a']} vs {row['variant_b']} & "
            f"{row['kendall_tau_test_mean']:.4f} "
            f"\\pm {row['kendall_tau_test_std']:.4f} \\\\"
        )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
