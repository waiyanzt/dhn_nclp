"""Build DHN bundles for WordNet KG link prediction.

The raw WordNet augmentation package has the same schema as the augmented
FB15k package:

    <variant>/entities.dict
    <variant>/relations.dict
    <variant>/data.txt

Unlike FB15k, WordNet p3 enumeration is tractable for these variants, so the
default pattern set is the vanilla DHN set {p1, c2, p3}.

Usage:
    python -m preprocess.wordnet.link_prediction --variant no_changes
    python -m preprocess.wordnet.link_prediction --variant no_changes,all_inverse_edges,transitive_edges
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhn.graph_enumerations import (  # noqa: E402
    cycle_mapping_index,
    path_mapping_index,
    single_node_mapping_index,
)


RAW_ROOT = "data/raw/wordnet_3hops_augmented_full"
OUT_DIR = "data/preprocessed"
SEED = 1566911444
PATTERNS = ["p1", "c2", "p3"]
VALID_VARIANTS = {"no_changes", "all_inverse_edges", "transitive_edges"}

PATTERN_FNS = {
    "p1": single_node_mapping_index,
    "c2": lambda g: cycle_mapping_index(g, length_bound=2),
    "p3": path_mapping_index,
}


def load_id_maps(raw_dir: Path):
    ent2id = {}
    with open(raw_dir / "entities.dict") as f:
        for line in f:
            eid, entity = line.rstrip("\n").split("\t")
            ent2id[entity] = int(eid)

    rel2id = {}
    with open(raw_dir / "relations.dict") as f:
        for line in f:
            rid, rel = line.rstrip("\n").split("\t")
            rel2id[rel] = int(rid)

    return ent2id, rel2id


def load_triples(raw_dir: Path, ent2id, rel2id):
    rows = []
    with open(raw_dir / "data.txt") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            rows.append((ent2id[h], rel2id[r], ent2id[t]))
    return np.asarray(rows, dtype=np.int64)


def stratified_split(triples, seed):
    """80/10/10 per relation. Relations with <5 triples go fully to train."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for rel_id in np.unique(triples[:, 1]):
        idx = np.where(triples[:, 1] == rel_id)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n < 5:
            train_idx.extend(idx.tolist())
            continue
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.10))
        n_train = n - n_val - n_test
        if n_train < 1:
            train_idx.extend(idx.tolist())
            continue
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    return (
        triples[sorted(train_idx)],
        triples[sorted(val_idx)],
        triples[sorted(test_idx)],
    )


def build_graph(train_triples, num_entities):
    """Undirected simple entity graph built from train triples only."""
    lo = np.minimum(train_triples[:, 0], train_triples[:, 2])
    hi = np.maximum(train_triples[:, 0], train_triples[:, 2])
    mask = lo != hi
    pairs = np.unique(np.stack([lo[mask], hi[mask]], axis=1), axis=0)

    nxg = nx.Graph()
    nxg.add_nodes_from(range(num_entities))
    nxg.add_edges_from(pairs.tolist())

    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    edge_index = torch.from_numpy(np.vstack([src, dst]))
    return nxg, edge_index


def preprocess_one(raw_root: Path, variant: str, out_dir: Path, seed: int) -> None:
    raw_dir = raw_root / variant
    if not raw_dir.is_dir():
        raise SystemExit(f"Missing WordNet variant directory: {raw_dir}")

    print(f"\n=== WordNet DHN-LP preprocess | variant={variant} ===", flush=True)
    ent2id, rel2id = load_id_maps(raw_dir)
    num_entities = len(ent2id)
    num_relations = len(rel2id)
    print(f"  Entities: {num_entities:,}  Relations: {num_relations:,}")

    triples = load_triples(raw_dir, ent2id, rel2id)
    print(f"  Total triples: {len(triples):,}")

    train, val, test = stratified_split(triples, seed)
    print(f"  Split: train={len(train):,} val={len(val):,} test={len(test):,}")

    nxg, edge_index = build_graph(train, num_entities)
    print(f"  Graph: nodes={nxg.number_of_nodes():,} undirected_edges={nxg.number_of_edges():,}")

    p3_rows_est = int(sum(d * d for _, d in nxg.degree()))
    print(f"  p3 rows estimate: {p3_rows_est:,}")

    print(f"  Enumerating patterns {PATTERNS}...")
    mapping_index_dict = {}
    for name in PATTERNS:
        mapping_index_dict.update(PATTERN_FNS[name](nxg))
    for k, v in mapping_index_dict.items():
        print(f"    {k}: {None if v is None else tuple(v.shape)}")

    data = Data(x=None, edge_index=edge_index, mapping_index_dict=mapping_index_dict)
    data.num_nodes = num_entities
    data.batch = torch.zeros(num_entities, dtype=torch.long)
    data.batch_size = 1

    all_triples = np.concatenate([train, val, test], axis=0)
    bundle = {
        "data": data,
        "splits": {
            "train": torch.from_numpy(train).long(),
            "val": torch.from_numpy(val).long(),
            "test": torch.from_numpy(test).long(),
            "all_triples": torch.from_numpy(all_triples).long(),
        },
        "meta": {
            "dataset": "wordnet_3hops_augmented_full",
            "dataset_slug": f"wordnet_{variant}",
            "variant": variant,
            "num_entities": num_entities,
            "num_relations": num_relations,
            "num_nodes_total": num_entities,
            "patterns": PATTERNS,
            "splits_seed": seed,
            "source": f"WordNet 3-hop augmented full/{variant}",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"WordNet3Hop_dhn_lp_{variant}.pt"
    torch.save(bundle, out_path)
    print(f"  Saved -> {out_path}")


def parse_variants(value: str) -> list[str]:
    vals = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in vals if v not in VALID_VARIANTS]
    if bad:
        raise SystemExit(f"Unknown variant(s): {bad}; expected one of {sorted(VALID_VARIANTS)}")
    return vals


def main():
    ap = argparse.ArgumentParser(description="WordNet DHN-LP preprocessing.")
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument(
        "--variant",
        default="no_changes",
        help="Comma-separated: no_changes, all_inverse_edges, transitive_edges",
    )
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    for variant in parse_variants(args.variant):
        preprocess_one(Path(args.raw_root), variant, Path(args.out_dir), args.seed)


if __name__ == "__main__":
    main()
