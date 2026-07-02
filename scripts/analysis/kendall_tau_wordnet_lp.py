"""Pairwise WordNet LP robustness from saved DHN candidate-score artifacts.

The score artifacts follow the lab RGCN protocol: 200 deterministic tail
candidates per shared test query, with the true tail in the final column.

Usage:
    python -m scripts.analysis.kendall_tau_wordnet_lp
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


VARIANTS = ("no_changes", "all_inverse_edges", "transitive_edges")
SEEDS = (1566911444, 20241017, 20251017)


def load_scores(root: Path, variant: str, seed: int) -> dict[str, np.ndarray]:
    path = root / f"kendall_tail_scores_wordnet_{variant}_seed{seed}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def finite_tau(a: np.ndarray, b: np.ndarray) -> float:
    value = kendalltau(a, b, variant="b", nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else float("nan")


def per_query_tau(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    values = np.asarray([
        finite_tau(a, b) for a, b in zip(scores_a, scores_b)
    ], dtype=np.float64)
    return float(np.nanmean(values))


def true_tail_ranks(scores: np.ndarray) -> np.ndarray:
    return 1 + (scores > scores[:, [-1]]).sum(axis=1)


def top1_match(candidate_ids: np.ndarray, scores_a: np.ndarray,
               scores_b: np.ndarray) -> float:
    row_ids = np.arange(len(candidate_ids))
    best_a = candidate_ids[row_ids, np.argmax(scores_a, axis=1)]
    best_b = candidate_ids[row_ids, np.argmax(scores_b, axis=1)]
    return float(np.mean(best_a == best_b))


def topk_tau(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    values = []
    for a, b in zip(scores_a, scores_b):
        top_a = np.argpartition(a, -k)[-k:]
        top_b = np.argpartition(b, -k)[-k:]
        union = np.union1d(top_a, top_b)
        if len(union) < 2:
            continue
        rank_a = rankdata(-a[union], method="average")
        rank_b = rankdata(-b[union], method="average")
        tau = finite_tau(rank_a, rank_b)
        if np.isfinite(tau):
            values.append(tau)
    return float(np.mean(values)) if values else float("nan")


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    for key in ("test_pos", "candidate_ids", "labels"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"WordNet Kendall artifacts differ in {key}")

    scores_a = a["scores"]
    scores_b = b["scores"]
    ranks_a = true_tail_ranks(scores_a)
    ranks_b = true_tail_ranks(scores_b)
    hit1_a, hit1_b = ranks_a <= 1, ranks_b <= 1
    hit3_a, hit3_b = ranks_a <= 3, ranks_b <= 3
    rho = spearmanr(ranks_a, ranks_b).statistic

    return {
        "overall_tau": finite_tau(scores_a.ravel(), scores_b.ravel()),
        "mrr_rank_spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "per_query_tau": per_query_tau(scores_a, scores_b),
        "top1_match": top1_match(a["candidate_ids"], scores_a, scores_b),
        "top3_tau": topk_tau(scores_a, scores_b, 3),
        "hit1_tau": finite_tau(hit1_a, hit1_b),
        "hit3_tau": finite_tau(hit3_a, hit3_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default="results/h100/wordnet_lp_baseline",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--output",
        default="reports/kendall_tau_wordnet_lp.csv",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    metric_names = (
        "overall_tau",
        "mrr_rank_spearman",
        "per_query_tau",
        "top1_match",
        "top3_tau",
        "hit1_tau",
        "hit3_tau",
    )

    for variant_a, variant_b in itertools.combinations(args.variants, 2):
        per_seed = {metric: [] for metric in metric_names}
        for seed in args.seeds:
            metrics = compare(
                load_scores(root, variant_a, seed),
                load_scores(root, variant_b, seed),
            )
            print(f"{variant_a} vs {variant_b} seed={seed}: "
                  f"tau={metrics['overall_tau']:.4f}")
            for metric, value in metrics.items():
                per_seed[metric].append(value)

        row = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "n_seeds": len(args.seeds),
        }
        for metric in metric_names:
            values = np.asarray(per_seed[metric], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            row[f"{metric}_std"] = (
                float(np.std(finite, ddof=0)) if len(finite) else float("nan")
            )
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary (mean +/- population std)")
    for row in rows:
        print(
            f"{row['variant_a']} vs {row['variant_b']}: "
            f"tau={row['overall_tau_mean']:.4f} +/- "
            f"{row['overall_tau_std']:.4f}, "
            f"H@1 tau={row['hit1_tau_mean']:.4f} +/- "
            f"{row['hit1_tau_std']:.4f}, "
            f"H@3 tau={row['hit3_tau_mean']:.4f} +/- "
            f"{row['hit3_tau_std']:.4f}"
        )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
