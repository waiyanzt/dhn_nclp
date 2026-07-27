"""Build leakage-free, four-variant WordNet bundles for joint DHN training.

This is the DHN counterpart of INV-GNN's
``preprocess_WORDNET_rgcn_augmentation.py``. It consumes the same raw package
and writes both the shared NPZ contract and one DHN graph bundle per variant.

Example:
    python -m preprocess.wordnet.augmentation \
      --data-dir ../INV-GNN/src/baselines/CMPNN/data/raw/wordnet_3hops_augmented_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data


VARIANTS = (
    "no_changes",
    "all_inverse_edges",
    "transitive_edges",
    "universal_edges",
)
TRANSITIVE_PREFIX = "__transitive__"
FORMAT_VERSION = "wordnet_lp_four_variants_v1"


def load_dict(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t", 1)
            if len(fields) != 2:
                raise ValueError(f"Bad dictionary line {line_number} in {path}")
            raw_id, name = fields
            if name in mapping:
                raise ValueError(f"Duplicate name {name!r} in {path}")
            mapping[name] = int(raw_id)
    return mapping


def validate_dense_ids(name: str, mapping: dict[str, int]) -> None:
    identifiers = sorted(mapping.values())
    if identifiers != list(range(len(mapping))):
        raise ValueError(f"{name} IDs in the raw package are not dense from zero")


def load_triples(
    path: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if fields == [""]:
                continue
            if len(fields) != 3:
                raise ValueError(f"Bad triple line {line_number} in {path}")
            head, relation, tail = fields
            try:
                rows.append(
                    (
                        entity_to_id[head],
                        relation_to_id[relation],
                        entity_to_id[tail],
                    )
                )
            except KeyError as exc:
                raise KeyError(
                    f"Unknown entity/relation {exc.args[0]!r} on line "
                    f"{line_number} of {path}"
                ) from exc
    return (
        np.asarray(rows, dtype=np.int32)
        if rows
        else np.empty((0, 3), dtype=np.int32)
    )


def triple_set(array: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in np.asarray(array)}


def ensure_unique(name: str, array: np.ndarray) -> None:
    duplicate_count = len(array) - len(triple_set(array))
    if duplicate_count:
        raise AssertionError(f"{name} contains {duplicate_count} duplicate triples")


def validate_splits(
    train_arrays: dict[str, np.ndarray],
    official_train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    relation_vocab: list[str],
    base_relation_ids: set[int],
) -> None:
    named_arrays = {
        "official_train": official_train,
        "validation": validation,
        "test": test,
        **{f"train_{name}": value for name, value in train_arrays.items()},
    }
    for name, array in named_arrays.items():
        ensure_unique(name, array)

    official_set = triple_set(official_train)
    validation_set = triple_set(validation)
    test_set = triple_set(test)
    heldout = validation_set | test_set
    if validation_set & test_set:
        raise AssertionError("Official validation and test splits overlap")
    if triple_set(train_arrays["no_changes"]) != official_set:
        raise AssertionError("no_changes/data.txt must exactly equal train.txt")

    for variant, array in train_arrays.items():
        values = triple_set(array)
        if official_set - values:
            raise AssertionError(f"{variant} omits official training triples")
        overlap = values & heldout
        if overlap:
            raise AssertionError(
                f"{variant} directly contains {len(overlap)} held-out triples"
            )

    for split_name, array in (("validation", validation), ("test", test)):
        derived = {int(r) for _, r, _ in array if int(r) not in base_relation_ids}
        if derived:
            raise AssertionError(f"{split_name} contains derived relation IDs")

    relation_to_id = {name: index for index, name in enumerate(relation_vocab)}
    inverse_by_base = {}
    for base_id in base_relation_ids:
        inverse_name = f"{relation_vocab[base_id]}__inv"
        if inverse_name in relation_to_id:
            inverse_by_base[base_id] = relation_to_id[inverse_name]
    heldout_inverses = {
        (int(t), inverse_by_base[int(r)], int(h))
        for h, r, t in np.concatenate([validation, test], axis=0)
        if int(r) in inverse_by_base
    }
    for variant in ("all_inverse_edges", "universal_edges"):
        overlap = triple_set(train_arrays[variant]) & heldout_inverses
        if overlap:
            raise AssertionError(
                f"{variant} contains {len(overlap)} inverses of held-out triples"
            )

    inverse_ids = {
        index
        for index, name in enumerate(relation_vocab)
        if name.endswith("__inv")
    }
    shortcut_ids = {
        index
        for index, name in enumerate(relation_vocab)
        if name.startswith(TRANSITIVE_PREFIX)
    }
    used = {
        variant: {int(r) for _, r, _ in triples}
        for variant, triples in train_arrays.items()
    }
    if used["no_changes"] - base_relation_ids:
        raise AssertionError("no_changes contains a derived relation")
    if used["all_inverse_edges"] - (base_relation_ids | inverse_ids):
        raise AssertionError("all_inverse_edges contains an unexpected relation")
    if used["transitive_edges"] & inverse_ids:
        raise AssertionError("transitive_edges contains inverse relations")
    if used["transitive_edges"] - (base_relation_ids | shortcut_ids):
        raise AssertionError("transitive_edges contains an unexpected relation")
    if used["universal_edges"] - (
        base_relation_ids | inverse_ids | shortcut_ids
    ):
        raise AssertionError("universal_edges contains an unexpected relation")


def category_counts(
    triples: np.ndarray, relation_vocab: list[str]
) -> dict[str, int]:
    counts = {"base": 0, "inverse": 0, "shortcut": 0}
    for _, relation, _ in triples:
        name = relation_vocab[int(relation)]
        if name.startswith(TRANSITIVE_PREFIX):
            counts["shortcut"] += 1
        elif name.endswith("__inv"):
            counts["inverse"] += 1
        else:
            counts["base"] += 1
    return counts


def read_augmentation_summary(
    data_dir: Path, variant: str, triple_count: int
) -> dict[str, Any]:
    path = data_dir / variant / "augmentation_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; the raw augmentation manifest is required"
        )
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("variant") != variant:
        raise AssertionError(
            f"{path} identifies variant {summary.get('variant')!r}, expected {variant!r}"
        )
    declared_count = summary.get("num_edges", summary.get("total_edges"))
    if declared_count is not None and int(declared_count) != triple_count:
        raise AssertionError(
            f"{path} declares {declared_count} edges but data.txt has {triple_count}"
        )
    return summary


def make_mapping_tensors(
    train_triples: np.ndarray, num_entities: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Build the same p1/c2 mappings as the legacy DHN WordNet preprocessor.

    Relation labels intentionally do not affect the DHN entity graph. Parallel
    KG edges collapse into one undirected entity pair, while DistMult retains
    and learns the shared relation vocabulary.
    """
    endpoints = np.asarray(train_triples[:, [0, 2]], dtype=np.int64)
    low = np.minimum(endpoints[:, 0], endpoints[:, 1])
    high = np.maximum(endpoints[:, 0], endpoints[:, 1])
    mask = low != high
    pairs = np.unique(np.stack([low[mask], high[mask]], axis=1), axis=0)
    if len(pairs):
        directed = np.concatenate([pairs[:, ::-1], pairs], axis=0)
    else:
        directed = np.empty((0, 2), dtype=np.int64)
    p1 = torch.arange(num_entities, dtype=torch.long).reshape(-1, 1)
    c2 = torch.from_numpy(directed).long()
    edge_index = c2.T.contiguous()
    return edge_index, {"p1": p1, "c2": c2}, int(len(pairs))


def save_bundle(
    *,
    output_dir: Path,
    variant: str,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    num_entities: int,
    num_relations: int,
    entity_vocab: list[str],
    relation_vocab: list[str],
    augmentation_summary: dict[str, Any],
) -> dict[str, Any]:
    edge_index, mappings, undirected_edges = make_mapping_tensors(
        train, num_entities
    )
    data = Data(x=None, edge_index=edge_index, mapping_index_dict=mappings)
    data.num_nodes = num_entities
    data.batch = torch.zeros(num_entities, dtype=torch.long)
    data.batch_size = 1
    all_triples = np.concatenate([train, validation, test], axis=0)
    bundle = {
        "data": data,
        "splits": {
            "train": torch.from_numpy(train.astype(np.int64, copy=False)),
            "val": torch.from_numpy(validation.astype(np.int64, copy=False)),
            "test": torch.from_numpy(test.astype(np.int64, copy=False)),
            "all_triples": torch.from_numpy(
                all_triples.astype(np.int64, copy=False)
            ),
        },
        "vocab": {
            "entities": entity_vocab,
            "relations": relation_vocab,
        },
        "meta": {
            "dataset": "wordnet_3hops_augmented_full",
            "dataset_slug": f"wordnet_{variant}",
            "variant": variant,
            "num_entities": num_entities,
            "num_relations": num_relations,
            "num_nodes_total": num_entities,
            "patterns": ["p1", "c2"],
            "training_triples": int(len(train)),
            "undirected_entity_edges": undirected_edges,
            "directed_c2_mappings": int(mappings["c2"].shape[0]),
            "split_protocol": "official_leakage_free_four_variant_wordnet_splits",
            "format_version": FORMAT_VERSION,
            "source": f"WordNet 3-hop augmented full/{variant}",
            "augmentation_summary": augmentation_summary,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"WordNet3Hop_dhn_lp_{variant}.pt"
    torch.save(bundle, output)
    return {
        "variant": variant,
        "bundle": str(output.resolve()),
        "training_triples": int(len(train)),
        "undirected_entity_edges": undirected_edges,
        "directed_c2_mappings": int(mappings["c2"].shape[0]),
    }


def preprocess(
    data_dir: Path,
    source_splits_dir: Path,
    splits_output: Path,
    bundle_output_dir: Path,
) -> dict[str, Any]:
    entity_to_id = load_dict(source_splits_dir / "entities.dict")
    base_relation_to_id = load_dict(source_splits_dir / "relations.dict")
    shared_relation_to_id = load_dict(data_dir / "shared_relations.dict")
    validate_dense_ids("entity", entity_to_id)
    validate_dense_ids("base relation", base_relation_to_id)
    validate_dense_ids("shared relation", shared_relation_to_id)

    entity_vocab = [""] * len(entity_to_id)
    for name, index in entity_to_id.items():
        entity_vocab[index] = name
    relation_vocab = [""] * len(shared_relation_to_id)
    for name, index in shared_relation_to_id.items():
        relation_vocab[index] = name
    base_names = [
        name
        for name, _ in sorted(base_relation_to_id.items(), key=lambda item: item[1])
    ]
    base_ids = {shared_relation_to_id[name] for name in base_names}

    official_train = load_triples(
        source_splits_dir / "train.txt", entity_to_id, shared_relation_to_id
    )
    validation = load_triples(
        source_splits_dir / "valid.txt", entity_to_id, shared_relation_to_id
    )
    test = load_triples(
        source_splits_dir / "test.txt", entity_to_id, shared_relation_to_id
    )
    train_arrays = {
        variant: load_triples(
            data_dir / variant / "data.txt",
            entity_to_id,
            shared_relation_to_id,
        )
        for variant in VARIANTS
    }
    validate_splits(
        train_arrays,
        official_train,
        validation,
        test,
        relation_vocab,
        base_ids,
    )
    summaries = {
        variant: read_augmentation_summary(
            data_dir, variant, len(train_arrays[variant])
        )
        for variant in VARIANTS
    }

    splits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        splits_output,
        **{
            f"train_pos_{variant}": train_arrays[variant]
            for variant in VARIANTS
        },
        val_pos=validation,
        test_pos=test,
        entity_vocab=np.asarray(entity_vocab),
        relation_vocab=np.asarray(relation_vocab),
        num_entities=np.asarray(len(entity_vocab)),
        num_relations=np.asarray(len(relation_vocab)),
        num_base_relations=np.asarray(len(base_ids)),
        base_relation_ids=np.asarray(sorted(base_ids), dtype=np.int32),
        variant_names=np.asarray(VARIANTS),
        format_version=np.asarray(FORMAT_VERSION),
    )

    bundles = [
        save_bundle(
            output_dir=bundle_output_dir,
            variant=variant,
            train=train_arrays[variant],
            validation=validation,
            test=test,
            num_entities=len(entity_vocab),
            num_relations=len(relation_vocab),
            entity_vocab=entity_vocab,
            relation_vocab=relation_vocab,
            augmentation_summary=summaries[variant],
        )
        for variant in VARIANTS
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "split_protocol": "official_leakage_free_four_variant_wordnet_splits",
        "splits_npz": str(splits_output.resolve()),
        "official_split_counts": {
            "train": int(len(official_train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "num_entities": len(entity_vocab),
        "num_base_relations": len(base_ids),
        "num_shared_relations": len(relation_vocab),
        "variant_edge_categories": {
            variant: category_counts(train_arrays[variant], relation_vocab)
            for variant in VARIANTS
        },
        "bundles": bundles,
        "leakage_checks_passed": True,
    }
    manifest_path = bundle_output_dir / "WordNet3Hop_dhn_augmentation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build four leakage-free WordNet DHN augmentation bundles"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-splits-dir", type=Path, default=None)
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=Path("data/preprocessed/WordNet_four_variant_splits.npz"),
    )
    parser.add_argument(
        "--bundle-output-dir", type=Path, default=Path("data/preprocessed")
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    source_dir = (
        args.source_splits_dir.resolve()
        if args.source_splits_dir
        else data_dir / "original_splits"
    )
    manifest = preprocess(
        data_dir,
        source_dir,
        args.splits_output.resolve(),
        args.bundle_output_dir.resolve(),
    )
    print(
        f"[OK] Wrote {len(manifest['bundles'])} DHN bundles and "
        f"{manifest['splits_npz']}"
    )


if __name__ == "__main__":
    main()
