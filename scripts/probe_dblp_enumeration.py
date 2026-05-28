"""Predict DHN p3 enumeration cost on the flattened DBLP graph WITHOUT building it.

p3 (path_mapping_index) materializes every length-2 walk; its row count is
exactly sum_v deg(v)^2 over the undirected graph. Terms are huge hubs, so this
probe reports that sum (and the per-type breakdown) to decide PAPER_KEEP_FRAC
and whether term nodes need capping/dropping — before risking an OOM.

Usage:
    uv run python scripts/probe_dblp_enumeration.py --frac 1.0
    uv run python scripts/probe_dblp_enumeration.py --frac 0.10 --drop-terms
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

RAW = "data/raw/DBLP/"
SEED = 1566911444


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--drop-terms", action="store_true")
    ap.add_argument("--cap-term", type=int, default=0, help="Max papers per term (0 = no cap)")
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    al = pd.read_csv(RAW + "author_label.txt", sep="\t",
                     names=["author_id", "label", "name"], header=None, encoding="utf-8")
    pa = pd.read_csv(RAW + "paper_author.txt", sep="\t",
                     names=["paper_id", "author_id"], header=None, encoding="utf-8")
    pc = pd.read_csv(RAW + "paper_conf.txt", sep="\t",
                     names=["paper_id", "conf_id"], header=None, encoding="utf-8")
    pt = pd.read_csv(RAW + "paper_term.txt", sep="\t",
                     names=["paper_id", "term_id"], header=None, encoding="utf-8")

    valid_authors = set(al["author_id"])
    pa = pa[pa["author_id"].isin(valid_authors)]
    valid_papers = set(pa["paper_id"])
    pc = pc[pc["paper_id"].isin(valid_papers)]
    pt = pt[pt["paper_id"].isin(valid_papers)]
    valid_papers = set(pc["paper_id"])
    pa = pa[pa["paper_id"].isin(valid_papers)]
    pt = pt[pt["paper_id"].isin(valid_papers)]

    paper_list = np.array(sorted(valid_papers))
    k = max(1, int(len(paper_list) * args.frac))
    keep = set(rng.choice(paper_list, size=k, replace=False))
    pa = pa[pa["paper_id"].isin(keep)]
    pc = pc[pc["paper_id"].isin(keep)]
    pt = pt[pt["paper_id"].isin(keep)]

    if args.cap_term > 0:
        pt = pt.groupby("term_id", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.cap_term), random_state=SEED))

    # paper -> area(s) via author labels
    a2r = al.set_index("author_id")["label"]
    pr = pa.assign(area=pa["author_id"].map(a2r))[["paper_id", "area"]].drop_duplicates().dropna()

    # train-only P-C: approximate with ~70% of papers (split is paper-disjoint)
    train_papers = set(pd.Series(sorted(keep)).sample(frac=0.70, random_state=SEED))
    pc_train = pc[pc["paper_id"].isin(train_papers)]

    # Build undirected degree (distinct neighbors) keyed by (type, id).
    nbrs = defaultdict(set)

    def link(a, b):
        nbrs[a].add(b)
        nbrs[b].add(a)

    for p, a in pa[["paper_id", "author_id"]].itertuples(index=False):
        link(("P", p), ("A", a))
    if not args.drop_terms:
        for p, t in pt[["paper_id", "term_id"]].itertuples(index=False):
            link(("P", p), ("T", t))
    for p, c in pc_train[["paper_id", "conf_id"]].itertuples(index=False):
        link(("P", p), ("C", c))
    for p, r in pr.itertuples(index=False):
        link(("P", p), ("R", int(r)))

    deg_sq_by_type = defaultdict(int)
    n_by_type = defaultdict(int)
    edges = 0
    for node, ns in nbrs.items():
        d = len(ns)
        deg_sq_by_type[node[0]] += d * d
        n_by_type[node[0]] += 1
        edges += d
    edges //= 2
    p3_rows = sum(deg_sq_by_type.values())

    print(f"\n=== frac={args.frac}  drop_terms={args.drop_terms}  cap_term={args.cap_term} ===")
    print(f"nodes: " + "  ".join(f"{t}={n_by_type[t]}" for t in ("A", "P", "T", "C", "R")))
    print(f"undirected edges (deduped): {edges:,}   -> c2 rows ~= {2*edges:,}")
    print(f"p3 rows = sum deg^2 = {p3_rows:,}")
    print("  per-type deg^2 contribution:")
    for t in ("A", "P", "T", "C", "R"):
        share = 100.0 * deg_sq_by_type[t] / p3_rows if p3_rows else 0
        print(f"    {t}: {deg_sq_by_type[t]:,}  ({share:.1f}%)")
    gb = p3_rows * 3 * 8 / 1e9
    print(f"p3 tensor est: {gb:.2f} GB (int64, 3 cols)")


if __name__ == "__main__":
    main()
