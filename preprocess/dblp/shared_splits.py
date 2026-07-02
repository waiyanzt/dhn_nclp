"""Create shared paper-disjoint splits for DBLP paper-venue prediction.

The default uses every eligible paper. ``--paper-keep-frac`` is retained only
for explicit scalability studies; production baseline runs should use 1.0.

Usage:
    python -m preprocess.dblp.shared_splits
    python -m preprocess.dblp.shared_splits --paper-keep-frac 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 1566911444
TEST_RATIO = 0.20
VAL_RATIO = 0.10
MIN_CONF = 0


def load_eligible_tables(raw_dir: Path, min_conf: int):
    author_labels = pd.read_csv(
        raw_dir / "author_label.txt",
        sep="\t",
        names=["author_id", "label", "author_name"],
        header=None,
        encoding="utf-8",
    )
    paper_author = pd.read_csv(
        raw_dir / "paper_author.txt",
        sep="\t",
        names=["paper_id", "author_id"],
        header=None,
        encoding="utf-8",
    )
    paper_conf = pd.read_csv(
        raw_dir / "paper_conf.txt",
        sep="\t",
        names=["paper_id", "conf_id"],
        header=None,
        encoding="utf-8",
    )
    paper_term = pd.read_csv(
        raw_dir / "paper_term.txt",
        sep="\t",
        names=["paper_id", "term_id"],
        header=None,
        encoding="utf-8",
    )

    valid_authors = set(author_labels["author_id"])
    paper_author = paper_author[
        paper_author["author_id"].isin(valid_authors)
    ].reset_index(drop=True)

    valid_papers = set(paper_author["paper_id"])
    paper_conf = paper_conf[
        paper_conf["paper_id"].isin(valid_papers)
    ].reset_index(drop=True)
    paper_term = paper_term[
        paper_term["paper_id"].isin(valid_papers)
    ].reset_index(drop=True)

    conf_counts = paper_conf["conf_id"].value_counts()
    kept_confs = conf_counts.loc[lambda counts: counts >= min_conf].index
    paper_conf = paper_conf[
        paper_conf["conf_id"].isin(kept_confs)
    ].reset_index(drop=True)

    valid_papers = set(paper_conf["paper_id"])
    paper_author = paper_author[
        paper_author["paper_id"].isin(valid_papers)
    ].reset_index(drop=True)
    paper_term = paper_term[
        paper_term["paper_id"].isin(valid_papers)
    ].reset_index(drop=True)
    return paper_author, paper_conf, paper_term, valid_papers


def choose_papers(valid_papers, keep_frac: float, seed: int) -> np.ndarray:
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"paper_keep_frac must be in (0, 1], got {keep_frac}")

    papers = np.asarray(sorted(valid_papers), dtype=np.int64)
    if keep_frac == 1.0:
        return papers

    keep_count = max(1, int(len(papers) * keep_frac))
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(papers, size=keep_count, replace=False))


def split_papers(papers: np.ndarray, seed: int):
    train, remainder = train_test_split(
        papers,
        test_size=TEST_RATIO + VAL_RATIO,
        random_state=seed,
        shuffle=True,
    )
    relative_test_ratio = TEST_RATIO / (TEST_RATIO + VAL_RATIO)
    val, test = train_test_split(
        remainder,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True,
    )
    return tuple(np.sort(part).astype(np.int64) for part in (train, val, test))


def positive_pairs(paper_conf: pd.DataFrame, papers: np.ndarray) -> np.ndarray:
    pairs = paper_conf[["paper_id", "conf_id"]].drop_duplicates()
    return pairs[pairs["paper_id"].isin(papers)].to_numpy(dtype=np.int64)


def negative_pairs(
    positive: np.ndarray,
    papers: np.ndarray,
    conferences: np.ndarray,
) -> np.ndarray:
    true_pairs = set(map(tuple, positive.tolist()))
    return np.asarray(
        [
            (int(paper), int(conf))
            for paper in papers
            for conf in conferences
            if (int(paper), int(conf)) not in true_pairs
        ],
        dtype=np.int64,
    ).reshape(-1, 2)


def generate_splits(
    raw_dir: Path,
    output_path: Path,
    paper_keep_frac: float,
    seed: int,
    min_conf: int,
) -> None:
    _pa, paper_conf, _pt, valid_papers = load_eligible_tables(raw_dir, min_conf)
    paper_subset = choose_papers(valid_papers, paper_keep_frac, seed)
    paper_conf = paper_conf[
        paper_conf["paper_id"].isin(paper_subset)
    ].reset_index(drop=True)

    papers_train, papers_val, papers_test = split_papers(paper_subset, seed)
    train_pos = positive_pairs(paper_conf, papers_train)
    val_pos = positive_pairs(paper_conf, papers_val)
    test_pos = positive_pairs(paper_conf, papers_test)

    conferences = np.asarray(
        sorted(paper_conf["conf_id"].unique()), dtype=np.int64
    )
    train_neg = negative_pairs(train_pos, papers_train, conferences)
    val_neg = negative_pairs(val_pos, papers_val, conferences)
    test_neg = negative_pairs(test_pos, papers_test, conferences)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
        train_neg=train_neg,
        val_neg=val_neg,
        test_neg=test_neg,
        paper_subset=paper_subset,
        papers_train=papers_train,
        papers_val=papers_val,
        papers_test=papers_test,
        paper_keep_frac=np.asarray(paper_keep_frac, dtype=np.float64),
        split_seed=np.asarray(seed, dtype=np.int64),
        split_version=np.asarray("paper_disjoint_v2"),
    )

    print(f"Eligible papers: {len(valid_papers):,}")
    print(
        f"Selected papers: {len(paper_subset):,} "
        f"({100.0 * paper_keep_frac:.1f}%)"
    )
    print(
        "Paper split: "
        f"train={len(papers_train):,} val={len(papers_val):,} "
        f"test={len(papers_test):,}"
    )
    print(
        "Positive pairs: "
        f"train={len(train_pos):,} val={len(val_pos):,} test={len(test_pos):,}"
    )
    print(f"Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create shared full-DBLP paper-venue splits."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/DBLP"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz"
        ),
    )
    parser.add_argument("--paper-keep-frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-conf", type=int, default=MIN_CONF)
    args = parser.parse_args()

    generate_splits(
        args.raw_dir,
        args.output,
        args.paper_keep_frac,
        args.seed,
        args.min_conf,
    )


if __name__ == "__main__":
    main()
