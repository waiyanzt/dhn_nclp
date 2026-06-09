"""Build a DHN bundle for FB15k-237 KG link prediction.

p3 DROPPED — 188M rows infeasible (scripts/probe_fb15k_enumeration.py).
Patterns: p1, c2 only.
Splits: 80/10/10 stratified by relation (seed 1566911444).
Graph: train triples only (undirected, relations collapsed) to avoid leakage.

Bundle stores raw (h, r, t) integer-id triples. DistMult scorer + full
entity-ranking eval live in train_lp_fb15k.py.

Usage:
    python preprocess/preprocess_fb15k_dhn_lp.py
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhn.graph_enumerations import cycle_mapping_index, single_node_mapping_index

RAW_DIR = (
    "data/raw/fb15k237_augmented_inverse"
    "/fb15k237_augmented_inverse/no_changes"
)
OUT_PATH = "data/preprocessed/FB15k237_dhn_lp.pt"
SEED = 1566911444
PATTERNS = ["p1", "c2"]


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DIR,
                    help="Path to the no_changes directory")
    ap.add_argument("--out-path", default=OUT_PATH)
    return ap.parse_args()

PATTERN_FNS = {
    "p1": single_node_mapping_index,
    "c2": lambda g: cycle_mapping_index(g, length_bound=2),
}


def load_id_maps(raw_dir):
    ent2id = {}
    with open(os.path.join(raw_dir, "entities.dict")) as f:
        for line in f:
            eid, mid = line.strip().split("\t")
            ent2id[mid] = int(eid)

    rel2id = {}
    with open(os.path.join(raw_dir, "relations.dict")) as f:
        for line in f:
            rid, rpath = line.strip().split("\t")
            rel2id[rpath] = int(rid)

    return ent2id, rel2id


def load_triples(raw_dir, ent2id, rel2id):
    rows = []
    with open(os.path.join(raw_dir, "data.txt")) as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            rows.append((ent2id[h], rel2id[r], ent2id[t]))
    return np.array(rows, dtype=np.int64)


def stratified_split(triples, seed):
    """80/10/10 per-relation split. Relations with <5 triples go fully to train."""
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
    train = triples[sorted(train_idx)]
    val = triples[sorted(val_idx)]
    test = triples[sorted(test_idx)]
    return train, val, test


def build_graph(train_triples, num_entities):
    """Undirected simple entity graph built from train triples (relations collapsed)."""
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


def main():
    args = _parse_args()
    raw_dir = args.raw_dir
    out_path = args.out_path

    print(f"Loading data from {raw_dir}...")
    ent2id, rel2id = load_id_maps(raw_dir)
    num_entities = len(ent2id)
    num_relations = len(rel2id)
    print(f"  Entities: {num_entities:,}  Relations: {num_relations}")

    triples = load_triples(raw_dir, ent2id, rel2id)
    print(f"  Total triples: {len(triples):,}")

    print("Creating 80/10/10 stratified split...")
    train, val, test = stratified_split(triples, SEED)
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print("Building entity graph (train triples only)...")
    nxg, edge_index = build_graph(train, num_entities)
    print(f"  Nodes: {nxg.number_of_nodes():,}  Edges (undirected): {nxg.number_of_edges():,}")

    print(f"Enumerating patterns {PATTERNS}...")
    mapping_index_dict = {}
    for name in PATTERNS:
        mapping_index_dict.update(PATTERN_FNS[name](nxg))
    for k, v in mapping_index_dict.items():
        size = None if v is None else tuple(v.shape)
        print(f"  {k}: {size}")

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
            "num_entities": num_entities,
            "num_relations": num_relations,
            "num_nodes_total": num_entities,
            "patterns": PATTERNS,
            "splits_seed": SEED,
            "source": "FB15k-237 no_changes",
        },
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(bundle, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
