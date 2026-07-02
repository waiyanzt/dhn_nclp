"""Build DHN bundles for featureless Freebase BOOK node classification.

The split follows the lab baseline: stratified 60/20/20 over labeled BOOK
nodes with seed 1566911444. The graph is treated as undirected and deduplicated,
matching baseline loaders that add reverse edges internally.

Only mappings rooted at labeled target nodes are materialized. For a one-layer
DHN this gives exactly the same target-node outputs as full-graph enumeration,
without storing mappings whose outputs never enter training or evaluation.

Freebase exact_2 has roughly 49.8 billion target-rooted p3 mappings, so this
pipeline consistently uses the tractable pattern set {p1, c2}.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


DEFAULT_VARIANTS = ("unchanged", "exact_2")
DEFAULT_SPLIT_SEED = 1566911444
BOOK_TYPE = 0
NUM_CLASSES = 8
EMBEDDING_DIM = 64
DEFAULT_MAX_C2_MAPPINGS = 20_000_000


def read_nodes(path: Path) -> np.ndarray:
    rows = []
    max_id = -1
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            node_id, node_type = int(parts[0]), int(parts[2])
            rows.append((node_id, node_type))
            max_id = max(max_id, node_id)

    node_types = np.full(max_id + 1, -1, dtype=np.int16)
    for node_id, node_type in rows:
        node_types[node_id] = node_type
    if np.any(node_types < 0):
        raise ValueError(f"{path} does not contain contiguous node IDs")
    return node_types


def read_labels(path: Path, node_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_ids, labels = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            node_id, declared_type, label = int(parts[0]), int(parts[2]), int(parts[3])
            if declared_type != BOOK_TYPE or node_types[node_id] != BOOK_TYPE:
                raise ValueError(f"Non-BOOK label row in {path}: {line.rstrip()}")
            if not 0 <= label < NUM_CLASSES:
                raise ValueError(f"Label {label} outside [0, {NUM_CLASSES})")
            node_ids.append(node_id)
            labels.append(label)

    ids = np.asarray(node_ids, dtype=np.int64)
    values = np.asarray(labels, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Duplicate labeled node IDs in {path}")
    return ids, values


def stratified_split(
    node_ids: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ids, rest_ids, _, rest_labels = train_test_split(
        node_ids,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=seed,
    )
    val_ids, test_ids = train_test_split(
        rest_ids,
        test_size=0.5,
        stratify=rest_labels,
        random_state=seed,
    )
    return np.sort(train_ids), np.sort(val_ids), np.sort(test_ids)


def rooted_edge_mappings(
    link_path: Path,
    target_ids: np.ndarray,
    max_mappings: int | None = DEFAULT_MAX_C2_MAPPINGS,
) -> torch.Tensor:
    """Return deduplicated oriented [target_root, neighbor] c2 mappings."""
    targets = set(target_ids.tolist())
    neighbors: dict[int, set[int]] = defaultdict(set)
    mapping_count = 0

    with link_path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            parts = line.split("\t", 3)
            source, target = int(parts[0]), int(parts[1])
            if source == target:
                continue
            if source in targets:
                previous = len(neighbors[source])
                neighbors[source].add(target)
                mapping_count += len(neighbors[source]) - previous
            if target in targets:
                previous = len(neighbors[target])
                neighbors[target].add(source)
                mapping_count += len(neighbors[target]) - previous
            if max_mappings is not None and mapping_count > max_mappings:
                raise RuntimeError(
                    f"{link_path} exceeds the configured rooted c2 limit "
                    f"({max_mappings:,} unique mappings) after scanning "
                    f"{row_number:,} edge rows. Refusing to truncate or sample "
                    "the baseline graph."
                )
            if row_number % 50_000_000 == 0:
                print(
                    f"    scanned {row_number:,} directed edges; "
                    f"rooted c2={mapping_count:,}",
                    flush=True,
                )

    mappings = torch.empty((mapping_count, 2), dtype=torch.long)
    offset = 0
    for root in sorted(targets):
        adjacent = sorted(neighbors.get(root, ()))
        end = offset + len(adjacent)
        if adjacent:
            mappings[offset:end, 0] = root
            mappings[offset:end, 1] = torch.as_tensor(adjacent, dtype=torch.long)
        offset = end
    return mappings


def masks_from_ids(
    num_nodes: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = []
    for ids in (train_ids, val_ids, test_ids):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(ids)] = True
        masks.append(mask)
    return tuple(masks)


def preprocess_variant(
    raw_root: Path,
    out_dir: Path,
    variant: str,
    split_seed: int,
    embedding_dim: int,
    max_c2_mappings: int | None,
) -> Path:
    variant_dir = raw_root / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"Missing Freebase variant: {variant_dir}")

    canonical_types = read_nodes(raw_root / "unchanged" / "node.dat")
    canonical_ids, canonical_labels = read_labels(
        raw_root / "unchanged" / "label.dat", canonical_types
    )
    node_types = read_nodes(variant_dir / "node.dat")
    target_ids, target_labels = read_labels(variant_dir / "label.dat", node_types)

    if not np.array_equal(node_types, canonical_types):
        raise ValueError(f"Node IDs/types differ in variant {variant}")
    canonical = dict(zip(canonical_ids.tolist(), canonical_labels.tolist()))
    current = dict(zip(target_ids.tolist(), target_labels.tolist()))
    if current != canonical:
        raise ValueError(f"Labeled nodes/classes differ in variant {variant}")

    train_ids, val_ids, test_ids = stratified_split(
        canonical_ids, canonical_labels, split_seed
    )
    num_nodes = len(node_types)
    train_mask, val_mask, test_mask = masks_from_ids(
        num_nodes, train_ids, val_ids, test_ids
    )

    print(f"\n=== Freebase DHN-NC preprocess | variant={variant} ===")
    print(
        f"  Nodes={num_nodes:,} labeled BOOK={len(canonical_ids):,} "
        f"classes declared={NUM_CLASSES}"
    )
    print(
        f"  Split train={len(train_ids):,} val={len(val_ids):,} "
        f"test={len(test_ids):,} seed={split_seed}"
    )
    print("  Scanning forward edges and building rooted c2 mappings...")
    c2 = rooted_edge_mappings(
        variant_dir / "link.dat",
        canonical_ids,
        max_mappings=max_c2_mappings,
    )
    p1 = torch.from_numpy(np.sort(canonical_ids)).long().unsqueeze(1)
    print(f"  p1={tuple(p1.shape)} c2={tuple(c2.shape)}")

    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[torch.from_numpy(canonical_ids)] = torch.from_numpy(canonical_labels)
    data = Data(
        x=None,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        mapping_index_dict={"p1": p1, "c2": c2},
        num_nodes=num_nodes,
    )
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    data.batch_size = 1

    payload = {
        "data": data,
        "num_features": embedding_dim,
        "num_classes": NUM_CLASSES,
        "meta": {
            "dataset": "Freebase",
            "variant": variant,
            "target_type": "BOOK",
            "target_type_id": BOOK_TYPE,
            "num_labeled_targets": len(canonical_ids),
            "observed_classes": sorted(np.unique(canonical_labels).tolist()),
            "declared_num_classes": NUM_CLASSES,
            "split_seed": split_seed,
            "split_protocol": "stratified_60_20_20",
            "patterns": ["p1", "c2"],
            "mapping_scope": "labeled_target_roots",
            "feature_protocol": "learned_node_embeddings",
            "p3_omission": "exact_2 target-rooted p3 estimate is ~49.8B rows",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"Freebase_dhn_nc_{variant}.pt"
    torch.save(payload, out_path)
    print(f"  Saved -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-root", default="data/raw/dataset_variant_3hops_filter"
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument("--out-dir", default="data/preprocessed")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument(
        "--max-c2-mappings",
        type=int,
        default=DEFAULT_MAX_C2_MAPPINGS,
        help=(
            "Abort rather than materialize more rooted c2 mappings; "
            "use 0 to disable the guard."
        ),
    )
    args = parser.parse_args()

    max_c2_mappings = (
        None if args.max_c2_mappings == 0 else args.max_c2_mappings
    )
    for variant in args.variants:
        preprocess_variant(
            Path(args.raw_root),
            Path(args.out_dir),
            variant,
            args.split_seed,
            args.embedding_dim,
            max_c2_mappings,
        )


if __name__ == "__main__":
    main()
