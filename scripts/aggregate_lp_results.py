"""Aggregate the per-(task, variant) lp_summary_*.csv files into one wide-format
summary table for easy reading and emailing.

Reads:  data/results/lp_summary_<task>_<variant>.csv
Writes: data/results/summary.csv

Run from repo root:
    uv run python scripts/aggregate_lp_results.py
"""
from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict

RESULTS_DIR = "data/results"
OUT_PATH = os.path.join(RESULTS_DIR, "summary.csv")

# Random baselines per task (1 true pos vs K randomly sampled negs).
# md/ml use K=19 (20 candidates), mg uses K=2 (3 candidates).
RANDOM = {
    "md": {"K": 19, "auc": 0.5000, "ap": 0.0500, "mrr": 0.1798,
           "hits@1": 0.0500, "hits@3": 0.1500, "hits@5": 0.2500},
    "mg": {"K": 2,  "auc": 0.5000, "ap": 0.3333, "mrr": 0.6111,
           "hits@1": 0.3333, "hits@3": 1.0000, "hits@5": 1.0000},
    "ml": {"K": 19, "auc": 0.5000, "ap": 0.0500, "mrr": 0.1798,
           "hits@1": 0.0500, "hits@3": 0.1500, "hits@5": 0.2500},
}

HEADLINE_METRICS = ["auc", "ap", "mrr", "hits@1", "hits@3", "hits@5", "best_epoch"]
PATTERN = re.compile(r"lp_summary_(md|mg|ml)_(v[1-4])\.csv$")


def load_one(path: str) -> tuple[str, str, dict]:
    """Return (task, variant, {metric: (mean, std)})."""
    m = PATTERN.search(path)
    if not m:
        raise ValueError(f"Bad filename: {path}")
    task, variant = m.group(1), m.group(2)
    out: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = (float(row["mean"]), float(row["std"]))
    return task, variant, out


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "lp_summary_*.csv")))
    if not files:
        raise SystemExit(f"No lp_summary_*.csv files found in {RESULTS_DIR}")

    rows_by_task: dict[str, list] = defaultdict(list)
    for f in files:
        task, variant, metrics = load_one(f)
        rows_by_task[task].append((variant, metrics))

    # Sort variants within each task
    for task in rows_by_task:
        rows_by_task[task].sort(key=lambda x: x[0])

    headers = (
        ["task", "variant", "K_negs"]
        + [f"{m}_mean" for m in HEADLINE_METRICS]
        + [f"{m}_std" for m in HEADLINE_METRICS]
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        # Self-documenting preamble (pandas can skip with comment="#")
        w.writerow(["# DHN-LP vanilla sweep summary (patterns: p1, c2, p3; no input features)"])
        w.writerow(["# Each row is mean / std across 3 seeds: 1566911444, 20241017, 20251017"])
        w.writerow(["# Random baseline rows are included per task for direct comparison."])
        w.writerow([])
        w.writerow(headers)

        for task in ("md", "mg", "ml"):
            if task not in rows_by_task:
                continue
            r = RANDOM[task]
            # Random baseline row first
            base = [task, "RANDOM", r["K"]]
            base += [fmt(r[m]) if m in r else "" for m in HEADLINE_METRICS]
            base += [fmt(0.0) for _ in HEADLINE_METRICS]  # zero std for the theoretical baseline
            w.writerow(base)
            # Actual results
            for variant, metrics in rows_by_task[task]:
                row = [task, variant, r["K"]]
                row += [fmt(metrics[m][0]) if m in metrics else "" for m in HEADLINE_METRICS]
                row += [fmt(metrics[m][1]) if m in metrics else "" for m in HEADLINE_METRICS]
                w.writerow(row)
            w.writerow([])  # blank line between tasks for readability

    print(f"Wrote {OUT_PATH}")
    print(f"  {sum(len(v) for v in rows_by_task.values())} (task, variant) rows")
    print(f"  + {len(rows_by_task)} random-baseline reference rows")


if __name__ == "__main__":
    main()
