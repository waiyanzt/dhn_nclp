"""Build DHN bundles for WordNet KG link prediction.

The raw WordNet augmentation package has the same schema as the augmented
FB15k package:

    <variant>/entities.dict
    <variant>/relations.dict
    <variant>/data.txt

The benchmark uses the same scalable two-layer {p1, c2} DHN specification as
the other heterogeneous-graph adaptations. This avoids dataset-specific use of
an explicitly materialized p3 mapping.

Splits follow the lab protocol: validation and test triples come from the
intersection shared by all variants, while each variant retains its additional
triples in training. Generate them first with:

    python -m preprocess.wordnet.shared_splits

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
    single_node_mapping_index,
)


RAW_ROOT = "data/raw/wordnet_3hops_augmented_full"
OUT_DIR = "data/preprocessed"
SEED = 1566911444
PATTERNS = ["p1", "c2"]
VALID_VARIANTS = {"no_changes", "all_inverse_edges", "transitive_edges"}

PATTERN_FNS = {
    "p1": single_node_mapping_index,
    "c2": lambda g: cycle_mapping_index(g, length_bound=2),
}


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


def preprocess_one(raw_root: Path, shared_splits: Path, variant: str,
                   out_dir: Path) -> None:
    raw_dir = raw_root / variant
    if not raw_dir.is_dir():
        raise SystemExit(f"Missing WordNet variant directory: {raw_dir}")

    print(f"\n=== WordNet DHN-LP preprocess | variant={variant} ===", flush=True)
    split_data = np.load(shared_splits)
    entity_vocab = split_data["entity_vocab"]
    relation_vocab = split_data["relation_vocab"]
    ent2id = {name: idx for idx, name in enumerate(entity_vocab.tolist())}
    rel2id = {name: idx for idx, name in enumerate(relation_vocab.tolist())}
    num_entities = int(split_data["num_entities"])
    num_relations = int(split_data["num_relations"])
    print(f"  Entities: {num_entities:,}  Relations: {num_relations:,}")

    train = split_data[f"train_pos_{variant}"].astype(np.int64, copy=False)
    val = split_data["val_pos"].astype(np.int64, copy=False)
    test = split_data["test_pos"].astype(np.int64, copy=False)
    print(f"  Shared split: train={len(train):,} val={len(val):,} test={len(test):,}")

    nxg, edge_index = build_graph(train, num_entities)
    print(f"  Graph: nodes={nxg.number_of_nodes():,} undirected_edges={nxg.number_of_edges():,}")

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
            "splits_seed": int(split_data.get("split_seed", SEED)),
            "split_protocol": "shared_intersection_80_10_10",
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
    ap.add_argument(
        "--shared-splits",
        default="data/preprocessed/WordNet_shared_splits.npz",
    )
    args = ap.parse_args()

    shared_splits = Path(args.shared_splits)
    if not shared_splits.is_file():
        raise SystemExit(
            f"Missing shared splits: {shared_splits}\n"
            "Run: python -m preprocess.wordnet.shared_splits"
        )
    for variant in parse_variants(args.variant):
        preprocess_one(
            Path(args.raw_root), shared_splits, variant, Path(args.out_dir)
        )


if __name__ == "__main__":
    main()
