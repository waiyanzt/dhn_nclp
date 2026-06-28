"""Create shared WordNet LP splits matching the lab baseline protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


VARIANTS = ("no_changes", "all_inverse_edges", "transitive_edges")
SEED = 1566911444


def load_dict(path: Path) -> dict[str, int]:
    mapping = {}
    with path.open() as f:
        for line in f:
            idx, name = line.rstrip("\n").split("\t", 1)
            mapping[name] = int(idx)
    return mapping


def load_triples(path: Path, entity2id: dict[str, int],
                 relation2id: dict[str, int]) -> np.ndarray:
    rows = []
    with path.open() as f:
        for line in f:
            head, relation, tail = line.rstrip("\n").split("\t")
            rows.append((entity2id[head], relation2id[relation], entity2id[tail]))
    return np.asarray(rows, dtype=np.int64)


def triple_keys(triples: np.ndarray, num_entities: int,
                num_relations: int) -> np.ndarray:
    return (
        triples[:, 0] * (num_relations * num_entities)
        + triples[:, 1] * num_entities
        + triples[:, 2]
    )


def build_shared_splits(raw_root: Path, out_path: Path, seed: int) -> None:
    entity2id = load_dict(raw_root / "no_changes" / "entities.dict")
    relation2id = load_dict(raw_root / "transitive_edges" / "relations.dict")
    num_entities = len(entity2id)
    num_relations = len(relation2id)

    triples = {
        variant: load_triples(
            raw_root / variant / "data.txt", entity2id, relation2id
        )
        for variant in VARIANTS
    }
    keys = {
        variant: triple_keys(values, num_entities, num_relations)
        for variant, values in triples.items()
    }
    key_sets = {variant: set(values.tolist()) for variant, values in keys.items()}

    base = triples["no_changes"]
    base_keys = keys["no_changes"]
    intersection_mask = np.fromiter(
        (
            int(key) in key_sets["all_inverse_edges"]
            and int(key) in key_sets["transitive_edges"]
            for key in base_keys
        ),
        dtype=bool,
        count=len(base_keys),
    )
    intersection = base[intersection_mask]
    if len(intersection) <= 50_000:
        raise ValueError(f"WordNet intersection unexpectedly small: {len(intersection)}")

    rng = np.random.default_rng(seed)
    intersection = intersection[rng.permutation(len(intersection))]
    n_val = int(len(intersection) * 0.1)
    n_test = int(len(intersection) * 0.1)
    n_train = len(intersection) - n_val - n_test
    train_intersection = intersection[:n_train]
    val_pos = intersection[n_train:n_train + n_val]
    test_pos = intersection[n_train + n_val:]

    base_key_set = key_sets["no_changes"]
    base_extra = base[~intersection_mask]
    train_splits = {"no_changes": np.concatenate(
        [train_intersection, base_extra], axis=0
    )}
    for variant in ("all_inverse_edges", "transitive_edges"):
        seen_base_mask = np.fromiter(
            (int(key) in key_sets[variant] for key in base_keys),
            dtype=bool,
            count=len(base_keys),
        )
        seen_base_extra = base[seen_base_mask & ~intersection_mask]
        variant_only_mask = np.fromiter(
            (int(key) not in base_key_set for key in keys[variant]),
            dtype=bool,
            count=len(keys[variant]),
        )
        variant_only = triples[variant][variant_only_mask]
        train_splits[variant] = np.concatenate(
            [train_intersection, seen_base_extra, variant_only], axis=0
        )

    entity_vocab = np.empty(num_entities, dtype=object)
    for name, idx in entity2id.items():
        entity_vocab[idx] = name
    relation_vocab = np.empty(num_relations, dtype=object)
    for name, idx in relation2id.items():
        relation_vocab[idx] = name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        val_pos=val_pos,
        test_pos=test_pos,
        train_pos_no_changes=train_splits["no_changes"],
        train_pos_all_inverse_edges=train_splits["all_inverse_edges"],
        train_pos_transitive_edges=train_splits["transitive_edges"],
        entity_vocab=entity_vocab.astype(str),
        relation_vocab=relation_vocab.astype(str),
        num_entities=np.asarray(num_entities),
        num_relations=np.asarray(num_relations),
        split_seed=np.asarray(seed),
    )

    print(f"Entities={num_entities:,} relations={num_relations:,}")
    print(f"Shared intersection={len(intersection):,}")
    print(f"Validation={len(val_pos):,} test={len(test_pos):,}")
    for variant in VARIANTS:
        print(f"Train {variant}={len(train_splits[variant]):,}")
    print(f"Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="data/raw/wordnet_3hops_augmented_full",
    )
    parser.add_argument(
        "--out-path",
        default="data/preprocessed/WordNet_shared_splits.npz",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build_shared_splits(Path(args.raw_root), Path(args.out_path), args.seed)


if __name__ == "__main__":
    main()
