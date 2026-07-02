"""Build baseline and invariant DHN bundles for IMDb node classification.

The four preprocessed variant directories share one node-id, label, split, and
feature contract. Baseline bundles retain their own adjacency. The `universal`
bundle uses the boolean union of all four adjacencies, implementing the IMDb*
graph-level mapping while leaving features and supervision unchanged.

Usage:
    python -m preprocess.imdb.node_classification \
      --variant v1,v2,v3,v4,universal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse
import torch
from torch_geometric.data import Data

from dhn.graph_enumerations import cycle_mapping_index, single_node_mapping_index


PATTERNS = ["p1", "c2"]
PATTERN_FNS = {
    "p1": lambda graph: single_node_mapping_index(graph),
    "c2": lambda graph: cycle_mapping_index(graph, length_bound=2),
}
VARIANT_DIR_NAMES = {
    "v1": "IMDB_preprocessed_star",
    "v2": "IMDB_preprocessed_star_t",
    "v3": "IMDB_preprocessed_star_t_2",
    "v4": "IMDB_preprocessed_star_t_3",
}
OUTPUT_NAMES = {
    "v1": "IMDB_dhn_nc.pt",
    "v2": "IMDB_dhn_nc_t.pt",
    "v3": "IMDB_dhn_nc_t_2.pt",
    "v4": "IMDB_dhn_nc_t_3.pt",
    "universal": "IMDB_dhn_nc_universal.pt",
}
SPLIT_KEYS = ("train_idx", "val_idx", "test_idx")


def variant_roots(preprocessed_root: Path) -> dict[str, Path]:
    return {
        variant: preprocessed_root / dirname
        for variant, dirname in VARIANT_DIR_NAMES.items()
    }


def load_contract(root: Path) -> dict:
    splits = np.load(root / "train_val_test_idx.npz")
    return {
        "node_types": np.load(root / "node_types.npy"),
        "labels": np.load(root / "labels.npy"),
        "splits": {key: splits[key] for key in SPLIT_KEYS},
    }


def validate_contract(roots: dict[str, Path]) -> dict:
    canonical = load_contract(roots["v1"])
    for variant, root in roots.items():
        current = load_contract(root)
        for key in ("node_types", "labels"):
            if not np.array_equal(canonical[key], current[key]):
                raise ValueError(f"{variant} has a different {key} contract")
        for split_key in SPLIT_KEYS:
            if not np.array_equal(
                canonical["splits"][split_key],
                current["splits"][split_key],
            ):
                raise ValueError(
                    f"{variant} has a different {split_key} split"
                )
    return canonical


def load_features(root: Path, node_types: np.ndarray) -> torch.Tensor:
    num_types = int(node_types.max()) + 1
    feature_parts = []
    feature_dim = None
    for node_type in range(num_types):
        part = scipy.sparse.load_npz(
            root / f"features_{node_type}.npz"
        ).toarray()
        if feature_dim is None:
            feature_dim = part.shape[1]
        elif part.shape[1] != feature_dim:
            raise ValueError(
                f"features_{node_type}.npz has dimension {part.shape[1]}, "
                f"expected {feature_dim}"
            )
        expected_rows = int((node_types == node_type).sum())
        if part.shape[0] != expected_rows:
            raise ValueError(
                f"features_{node_type}.npz has {part.shape[0]} rows, "
                f"expected {expected_rows}"
            )
        feature_parts.append(part)

    features = np.zeros((len(node_types), feature_dim), dtype=np.float32)
    for node_type, part in enumerate(feature_parts):
        features[node_types == node_type] = part
    return torch.from_numpy(features)


def load_adjacency(
    variant: str, roots: dict[str, Path]
) -> scipy.sparse.csr_matrix:
    if variant != "universal":
        return scipy.sparse.load_npz(roots[variant] / "adjM.npz").tocsr()

    adjacency = None
    for source_variant in ("v1", "v2", "v3", "v4"):
        current = scipy.sparse.load_npz(
            roots[source_variant] / "adjM.npz"
        ).tocsr().astype(bool)
        adjacency = (
            current.copy()
            if adjacency is None
            else adjacency.maximum(current)
        )
    return adjacency.astype(np.int8)


def build_bundle(
    variant: str,
    roots: dict[str, Path],
    feature_root: Path,
) -> dict:
    contract = validate_contract(roots)
    node_types = contract["node_types"]
    labels = contract["labels"]
    splits = contract["splits"]
    adjacency = load_adjacency(variant, roots).tocoo()

    num_nodes = len(node_types)
    movie_indices = np.where(node_types == 0)[0]
    if len(labels) != len(movie_indices):
        raise ValueError(
            f"label/movie mismatch: {len(labels)} labels and "
            f"{len(movie_indices)} movie nodes"
        )

    features = load_features(feature_root, node_types)
    targets = torch.full((num_nodes,), -1, dtype=torch.long)
    targets[movie_indices] = torch.from_numpy(labels).long()

    masks = {}
    for split_key, mask_name in (
        ("train_idx", "train_mask"),
        ("val_idx", "val_mask"),
        ("test_idx", "test_mask"),
    ):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[movie_indices[splits[split_key]]] = True
        masks[mask_name] = mask

    edge_index = torch.from_numpy(
        np.vstack([adjacency.row, adjacency.col])
    ).long()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (int(src), int(dst))
        for src, dst in zip(adjacency.row, adjacency.col)
        if src != dst
    )

    mapping_index_dict = {}
    for pattern in PATTERNS:
        mapping_index_dict.update(PATTERN_FNS[pattern](graph))

    data = Data(
        x=features,
        edge_index=edge_index,
        y=targets,
        mapping_index_dict=mapping_index_dict,
        **masks,
    )
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    data.batch_size = 1

    return {
        "data": data,
        "num_features": int(features.shape[1]),
        "num_classes": int(labels.max()) + 1,
        "meta": {
            "task": "node_classification",
            "variant": variant,
            "mapping_mode": (
                "invariant_union_graph"
                if variant == "universal"
                else "baseline"
            ),
            "union_of_variants": (
                ["v1", "v2", "v3", "v4"]
                if variant == "universal"
                else []
            ),
            "feature_source": str(feature_root),
            "patterns": list(PATTERNS),
            "num_nodes": num_nodes,
            "num_directed_edges": int(adjacency.nnz),
        },
    }


def parse_variants(value: str) -> list[str]:
    variants = [
        item.strip().lower() for item in value.split(",") if item.strip()
    ]
    allowed = {*VARIANT_DIR_NAMES, "universal"}
    unknown = [variant for variant in variants if variant not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variants {unknown}; expected a subset of {sorted(allowed)}"
        )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--variant",
        type=parse_variants,
        default=parse_variants("v1"),
        help="Comma-separated subset of v1,v2,v3,v4,universal.",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data/preprocessed"),
    )
    parser.add_argument(
        "--feature-variant",
        choices=sorted(VARIANT_DIR_NAMES),
        default="v2",
        help=(
            "Canonical topology-independent feature source. v2-v4 are "
            "identical; default: v2."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/preprocessed"),
    )
    args = parser.parse_args()

    roots = variant_roots(args.preprocessed_root)
    missing = [str(root) for root in roots.values() if not root.is_dir()]
    if missing:
        raise SystemExit(f"Missing IMDb preprocessed directories: {missing}")
    feature_root = roots[args.feature_variant]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variant:
        print(f"\n=== IMDb DHN-NC preprocess | variant={variant} ===")
        bundle = build_bundle(variant, roots, feature_root)
        data = bundle["data"]
        print(
            f"  nodes={data.num_nodes} directed_edges={data.edge_index.shape[1]} "
            f"features={bundle['num_features']} classes={bundle['num_classes']}"
        )
        print(
            f"  splits: train={int(data.train_mask.sum())} "
            f"val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}"
        )
        for name, mapping in data.mapping_index_dict.items():
            print(f"  {name}: {tuple(mapping.shape)}")

        output = args.out_dir / OUTPUT_NAMES[variant]
        torch.save(bundle, output)
        print(f"  Saved -> {output}")


if __name__ == "__main__":
    main()
