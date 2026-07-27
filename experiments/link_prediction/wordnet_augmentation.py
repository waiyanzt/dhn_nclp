"""Joint three-variant WordNet augmentation for DHN + DistMult.

The experiment contract mirrors INV-GNN's WordNet RGCN augmentation runner.
By default the shared model rotates through ``no_changes``,
``all_inverse_edges``, and ``universal_edges``.  The transitive-only arm remains
in the canonical preprocessing archive so fixed filtering is comparable with
RGCN, but it is not trained or evaluated unless explicitly requested.

* one model and optimizer are shared by every selected graph variant;
* one super-epoch visits every variant in a seeded random order;
* validation selects a checkpoint by mean filtered MRR across variants;
* legacy per-variant and shared-candidate invariance metrics are both emitted;
* exact super-epoch resume and CPU/CUDA peak-memory accounting are included.

The DHN architecture and loss remain the ones used by ``wordnet.py``.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from dhn.augmentation_utils import (
    EarlyStopper,
    atomic_torch_save,
    atomic_write_csv,
    checkpoint_size_bytes,
    cuda_memory_stats,
    link_binary_metrics,
    load_latest_training_state,
    mean_dict,
    merge_cuda_memory_stats,
    model_memory_bytes,
    process_peak_rss_bytes,
    reset_cuda_peak,
    resolve_device,
    save_latest_training_state,
    set_determinism,
    torch_load_full,
    triple_link_invariance_rows,
    write_json,
)
from dhn.utils import get_act_module, get_optimizer
from experiments.link_prediction.imdb import resolve_layers_config
from experiments.link_prediction.wordnet import DHNWordNetLP


VARIANTS = (
    "no_changes",
    "all_inverse_edges",
    "transitive_edges",
    "universal_edges",
)
DEFAULT_VARIANTS = (
    "no_changes",
    "all_inverse_edges",
    "universal_edges",
)
VARIANT_ALIASES = {
    "unchanged": "no_changes",
    "inverse": "all_inverse_edges",
    "transitive": "transitive_edges",
    "universal": "universal_edges",
}
FORMAT_VERSION = "wordnet_lp_four_variants_v1"


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Expected a nonempty comma-separated list without duplicates")
    return values


def canonicalize_variants(value: str) -> list[str]:
    variants = [VARIANT_ALIASES.get(item, item) for item in parse_csv(value)]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown WordNet variants: {unknown}")
    if len(variants) != len(set(variants)):
        raise ValueError("Variant aliases collapse to duplicate variants")
    return variants


def load_shared_splits(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        shared = {key: archive[key] for key in archive.files}
    required = {
        "val_pos",
        "test_pos",
        "entity_vocab",
        "relation_vocab",
        "num_entities",
        "num_relations",
        "base_relation_ids",
        *{f"train_pos_{variant}" for variant in VARIANTS},
    }
    missing = sorted(required - set(shared))
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    version = str(np.asarray(shared.get("format_version", "")).item())
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported split format {version!r}")
    archive_variants = tuple(
        str(item) for item in np.asarray(shared["variant_names"]).tolist()
    )
    if archive_variants != VARIANTS:
        raise ValueError(f"Unexpected variant order: {archive_variants}")
    return shared


def _triples(array: Any) -> np.ndarray:
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    return np.asarray(array, dtype=np.int64)


def _triple_set(array: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in np.asarray(array)}


def validate_shared_splits(shared: Mapping[str, np.ndarray]) -> None:
    num_entities = int(shared["num_entities"])
    num_relations = int(shared["num_relations"])
    relation_vocab = [str(value) for value in shared["relation_vocab"].tolist()]
    if len(relation_vocab) != num_relations:
        raise ValueError("relation_vocab length does not match num_relations")
    validation = _triples(shared["val_pos"])
    test = _triples(shared["test_pos"])
    heldout = _triple_set(validation) | _triple_set(test)
    if _triple_set(validation) & _triple_set(test):
        raise ValueError("Validation and test positives overlap")
    for variant in VARIANTS:
        train = _triples(shared[f"train_pos_{variant}"])
        if train.ndim != 2 or train.shape[1] != 3 or not len(train):
            raise ValueError(f"train_pos_{variant} is not a nonempty (N, 3) array")
        if train[:, [0, 2]].min() < 0 or train[:, [0, 2]].max() >= num_entities:
            raise ValueError(f"{variant} contains an invalid entity ID")
        if train[:, 1].min() < 0 or train[:, 1].max() >= num_relations:
            raise ValueError(f"{variant} contains an invalid relation ID")
        overlap = _triple_set(train) & heldout
        if overlap:
            raise ValueError(f"{variant} contains {len(overlap)} held-out triples")


def load_bundle(
    bundle_root: Path, pattern: str, variant: str
) -> dict[str, Any]:
    path = bundle_root / pattern.format(variant=variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m preprocess.wordnet.augmentation first."
        )
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    bundle["_path"] = str(path.resolve())
    return bundle


def prepare_bundles(
    splits_path: Path,
    bundle_root: Path,
    bundle_pattern: str,
    variants: list[str],
    fixed_negatives: int,
    candidate_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    shared = load_shared_splits(splits_path)
    validate_shared_splits(shared)
    num_entities = int(shared["num_entities"])
    num_relations = int(shared["num_relations"])
    entity_vocab = [str(value) for value in shared["entity_vocab"].tolist()]
    relation_vocab = [str(value) for value in shared["relation_vocab"].tolist()]

    bundles = {}
    for variant in variants:
        bundle = load_bundle(bundle_root, bundle_pattern, variant)
        meta = bundle["meta"]
        if meta.get("variant") != variant:
            raise ValueError(f"Bundle for {variant} identifies as {meta.get('variant')}")
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{variant} has unsupported bundle format")
        if int(meta["num_entities"]) != num_entities:
            raise ValueError(f"num_entities differs for {variant}")
        if int(meta["num_relations"]) != num_relations:
            raise ValueError(f"num_relations differs for {variant}")
        if bundle.get("vocab", {}).get("entities") != entity_vocab:
            raise ValueError(f"entity vocabulary differs for {variant}")
        if bundle.get("vocab", {}).get("relations") != relation_vocab:
            raise ValueError(f"relation vocabulary differs for {variant}")
        expected = {
            "train": _triples(shared[f"train_pos_{variant}"]),
            "val": _triples(shared["val_pos"]),
            "test": _triples(shared["test_pos"]),
        }
        for split_name, expected_array in expected.items():
            actual = _triples(bundle["splits"][split_name])
            if not np.array_equal(actual, expected_array):
                raise ValueError(f"{split_name} triples differ for {variant}")
        bundle["train_pos"] = expected["train"]
        bundle["edge_count"] = int(bundle["data"].edge_index.shape[1])
        bundles[variant] = bundle

    all_known = [
        _triples(shared[f"train_pos_{variant}"]) for variant in VARIANTS
    ] + [_triples(shared["val_pos"]), _triples(shared["test_pos"])]
    known_union = np.unique(np.concatenate(all_known, axis=0), axis=0)
    known_keys = set(
        pack_keys(known_union, num_entities, num_relations).tolist()
    )
    validation_candidates = fixed_tail_candidates(
        _triples(shared["val_pos"]),
        known_keys,
        num_entities,
        num_relations,
        fixed_negatives,
        candidate_seed + 101,
    )
    test_candidates = fixed_tail_candidates(
        _triples(shared["test_pos"]),
        known_keys,
        num_entities,
        num_relations,
        fixed_negatives,
        candidate_seed + 103,
    )
    prepared = dict(shared)
    prepared.update(
        {
            "known_union": known_union,
            "val_candidates": validation_candidates,
            "test_candidates": test_candidates,
        }
    )
    return bundles, prepared


def validate_model_bundle_contract(
    config: Mapping[str, Any], bundles: Mapping[str, Mapping[str, Any]]
) -> None:
    reference = next(iter(bundles.values()))
    patterns = tuple(reference["meta"].get("patterns", ()))
    if not patterns:
        raise ValueError("WordNet bundle does not declare meta.patterns")
    num_nodes = int(reference["meta"]["num_entities"])
    for variant, bundle in bundles.items():
        data = bundle["data"]
        if int(data.num_nodes) != num_nodes:
            raise ValueError(f"num_nodes differs for {variant}")
        if tuple(bundle["meta"].get("patterns", ())) != patterns:
            raise ValueError(f"pattern set differs for {variant}")
        if set(data.mapping_index_dict) != set(patterns):
            raise ValueError(
                f"{variant} mapping keys {sorted(data.mapping_index_dict)} "
                f"do not match declared patterns {sorted(patterns)}"
            )
    layers = config["model"].get("layers_config", ())
    if not layers:
        raise ValueError("model.layers_config must contain at least one layer")
    for layer_number, layer in enumerate(layers, start=1):
        if set(layer) != set(patterns):
            raise ValueError(
                f"Layer {layer_number} kernels {sorted(layer)} do not match "
                f"bundle patterns {sorted(patterns)}"
            )


def pack_keys(
    triples: np.ndarray, num_entities: int, num_relations: int
) -> np.ndarray:
    triples = np.asarray(triples, dtype=np.int64)
    return (
        triples[:, 0] * (num_relations * num_entities)
        + triples[:, 1] * num_entities
        + triples[:, 2]
    )


def fixed_tail_candidates(
    positives: np.ndarray,
    known_keys: set[int],
    num_entities: int,
    num_relations: int,
    negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if negatives_per_positive < 1:
        raise ValueError("fixed_negatives must be >= 1")
    rng = np.random.default_rng(seed)
    rows, labels, query_ids = [], [], []
    for query_id, (head, relation, true_tail) in enumerate(positives.tolist()):
        rows.append((head, relation, true_tail))
        labels.append(1)
        query_ids.append(query_id)
        selected: set[int] = set()
        attempts = 0
        while len(selected) < negatives_per_positive:
            candidate = int(rng.integers(0, num_entities))
            attempts += 1
            key = (
                head * (num_relations * num_entities)
                + relation * num_entities
                + candidate
            )
            if (
                candidate != true_tail
                and candidate not in selected
                and key not in known_keys
            ):
                selected.add(candidate)
            if attempts > max(1000, negatives_per_positive * 1000):
                raise RuntimeError(f"Could not sample candidates for query {query_id}")
        for candidate in sorted(selected):
            rows.append((head, relation, candidate))
            labels.append(0)
            query_ids.append(query_id)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(query_ids, dtype=np.int64),
    )


def move_graph_to_device(data, device: torch.device):
    data = data.to(device)
    data.mapping_index_dict = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in data.mapping_index_dict.items()
    }
    return data


def sample_negative_triples(
    positives: np.ndarray,
    num_entities: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = len(positives) * count
    heads = np.repeat(positives[:, 0], count)
    relations = np.repeat(positives[:, 1], count)
    tails = np.repeat(positives[:, 2], count)
    replacements = rng.integers(0, num_entities, size=total, dtype=np.int64)
    corrupt_head = rng.random(total) < 0.5
    heads = np.where(corrupt_head, replacements, heads)
    tails = np.where(corrupt_head, tails, replacements)
    return np.stack([heads, relations, tails], axis=1)


def score_triples(model, embeddings, triples: torch.Tensor) -> torch.Tensor:
    return model.score(
        embeddings[triples[:, 0]],
        triples[:, 1],
        embeddings[triples[:, 2]],
    )


def build_filter_dicts(*arrays: np.ndarray):
    tails: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    heads: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for array in arrays:
        for head, relation, tail in np.asarray(array).tolist():
            tails[(head, relation)].add(tail)
            heads[(relation, tail)].add(head)
    return tails, heads


@torch.no_grad()
def filtered_evaluate(
    model,
    graph,
    positives: np.ndarray,
    filters,
    device: torch.device,
    eval_batch_size: int,
    binary_k: int,
) -> dict[str, float]:
    model.eval()
    embeddings = model.encode(graph)
    tail_filters, head_filters = filters
    tail_ranks, head_ranks = [], []
    binary_scores, binary_labels = [], []
    binary_rng = np.random.default_rng(42)
    positive_tensor = torch.from_numpy(positives).long().to(device)

    for start in range(0, len(positives), eval_batch_size):
        batch = positive_tensor[start : start + eval_batch_size]
        heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]
        relation_embeddings = model.rel_emb(relations)
        tail_scores = ((embeddings[heads] * relation_embeddings) @ embeddings.T).cpu()
        head_scores = ((relation_embeddings * embeddings[tails]) @ embeddings.T).cpu()
        for row, (head, relation, tail) in enumerate(
            positives[start : start + len(batch)].tolist()
        ):
            true_tail_score = float(tail_scores[row, tail])
            true_head_score = float(head_scores[row, head])
            filtered_tails = tail_scores[row].clone()
            for other in tail_filters.get((head, relation), ()):
                if other != tail:
                    filtered_tails[other] = float("-inf")
            filtered_heads = head_scores[row].clone()
            for other in head_filters.get((relation, tail), ()):
                if other != head:
                    filtered_heads[other] = float("-inf")
            tail_ranks.append(int((filtered_tails >= true_tail_score).sum()))
            head_ranks.append(int((filtered_heads >= true_head_score).sum()))

            if binary_k:
                negatives: list[int] = []
                while len(negatives) < binary_k:
                    candidates = binary_rng.integers(
                        0, model.num_entities, size=binary_k * 4
                    )
                    for candidate in candidates.tolist():
                        if (
                            candidate not in tail_filters.get((head, relation), ())
                            and len(negatives) < binary_k
                        ):
                            negatives.append(candidate)
                binary_scores.append(true_tail_score)
                binary_scores.extend(tail_scores[row, negatives].tolist())
                binary_labels.append(1)
                binary_labels.extend([0] * binary_k)

    ranks = np.asarray(tail_ranks + head_ranks, dtype=np.float64)
    metrics = {
        "filtered_MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
    }
    if binary_k:
        probabilities = 1.0 / (
            1.0 + np.exp(-np.asarray(binary_scores, dtype=np.float64))
        )
        metrics.update(
            {
                f"binary_{key}": value
                for key, value in link_binary_metrics(
                    np.asarray(binary_labels, dtype=np.int64),
                    probabilities,
                    0.5,
                ).items()
            }
        )
    # DHNWordNetLP.encode assigns a generated embedding activation to graph.x.
    # It is rebuilt on every encode, so do not retain one activation per graph.
    graph.x = None
    return metrics


@torch.no_grad()
def shared_candidate_frame(
    model,
    graph,
    candidates,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    triples, labels, query_ids = candidates
    model.eval()
    embeddings = model.encode(graph)
    logits = []
    for start in range(0, len(triples), batch_size):
        batch = torch.from_numpy(triples[start : start + batch_size]).long().to(
            device
        )
        logits.append(score_triples(model, embeddings, batch).cpu())
    logits_array = torch.cat(logits).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits_array))
    bce = float(
        F.binary_cross_entropy_with_logits(
            torch.from_numpy(logits_array),
            torch.from_numpy(labels.astype(np.float32)),
        )
    )
    metrics = {
        "candidate_BCE": bce,
        **link_binary_metrics(labels, probabilities, 0.5),
    }
    frame = pd.DataFrame(
        {
            "query_id": query_ids,
            "head": triples[:, 0],
            "relation": triples[:, 1],
            "tail": triples[:, 2],
            "label": labels,
            "logit": logits_array,
            "score": probabilities,
        }
    )
    graph.x = None
    return metrics, frame


def resolve_arg(value, fallback):
    return fallback if value is None else value


def run_seed(
    args,
    config: Mapping[str, Any],
    variants: list[str],
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    set_determinism(seed)
    device = resolve_device(args.device or config.get("device", "auto"))
    bundles, shared = prepare_bundles(
        args.splits_npz,
        args.bundle_root,
        args.bundle_pattern,
        variants,
        args.fixed_negatives,
        args.candidate_seed,
    )
    validate_model_bundle_contract(config, bundles)
    num_entities = int(shared["num_entities"])
    num_relations = int(shared["num_relations"])

    model_config = config["model"]
    in_dim = int(model_config["in_dim"])
    layers_config = resolve_layers_config(model_config["layers_config"], in_dim)
    activation = model_config["activation"]
    model = DHNWordNetLP(
        num_entities=num_entities,
        num_relations=num_relations,
        in_dim=in_dim,
        layers_config=layers_config,
        act_module=get_act_module(activation["name"]),
        mapping_chunk_size=model_config.get("mapping_chunk_size"),
        checkpoint_chunks=model_config.get("checkpoint_chunks", True),
        **activation.get("kwargs", {}),
    ).to(device)
    optimizer_config = config["training"]["optimizer"]
    optimizer_kwargs = dict(optimizer_config.get("kwargs", {}))
    if args.lr is not None:
        optimizer_kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        optimizer_kwargs["weight_decay"] = args.weight_decay
    optimizer = get_optimizer(optimizer_config["name"])(
        model.parameters(), **optimizer_kwargs
    )

    super_epochs = resolve_arg(
        args.super_epochs, config["training"].get("super_epochs", 200)
    )
    patience = resolve_arg(args.patience, config["training"].get("patience", 10))
    eval_interval = resolve_arg(
        args.eval_interval, config["training"].get("eval_interval", 1)
    )
    batch_size_arg = resolve_arg(
        args.batch_size, config["training"].get("batch_size", 0)
    )
    neg_per_pos = resolve_arg(
        args.neg_per_pos, config["training"].get("neg_per_pos", 10)
    )
    grad_clip = resolve_arg(
        args.grad_clip, config["training"].get("grad_clip", 0.0)
    )
    eval_config = config["eval"]
    eval_batch_size = resolve_arg(
        args.eval_batch_size, eval_config.get("eval_batch_size", 256)
    )
    binary_k = int(eval_config.get("binary_k", 50))
    patience_checks = (
        args.patience_evals
        if args.patience_evals is not None
        else int(np.ceil(patience / eval_interval))
    )
    batch_sizes = {
        variant: (
            len(bundle["train_pos"])
            if batch_size_arg <= 0
            else min(batch_size_arg, len(bundle["train_pos"]))
        )
        for variant, bundle in bundles.items()
    }
    batches_per_variant = {
        variant: int(np.ceil(len(bundle["train_pos"]) / batch_sizes[variant]))
        for variant, bundle in bundles.items()
    }
    updates_per_super_epoch = sum(batches_per_variant.values())
    graphs = {
        variant: move_graph_to_device(bundle["data"], device)
        for variant, bundle in bundles.items()
    }
    filters = {
        variant: build_filter_dicts(
            bundle["train_pos"], shared["val_pos"], shared["test_pos"]
        )
        for variant, bundle in bundles.items()
    }

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    early_stopper = EarlyStopper("max", patience_checks, min_delta=0.0)
    rng = np.random.default_rng(seed)
    run_config = {
        "dataset": "WORDNET",
        "model": "DHNWordNetLP",
        "seed": seed,
        "variants": variants,
        "model_config": model_config,
        "optimizer": optimizer_config["name"],
        "optimizer_kwargs": optimizer_kwargs,
        "grad_clip": grad_clip,
        "batch_size": batch_size_arg,
        "neg_per_pos": neg_per_pos,
        "patience": patience,
        "patience_checks": patience_checks,
        "eval_interval": eval_interval,
        "eval_batch_size": eval_batch_size,
        "binary_k": binary_k,
        "score_batch_size": args.score_batch_size,
        "fixed_negatives": args.fixed_negatives,
        "candidate_seed": args.candidate_seed,
        "splits_npz": str(args.splits_npz.resolve()),
        "bundle_paths": {
            variant: bundles[variant]["_path"] for variant in variants
        },
    }

    history: list[dict[str, Any]] = []
    super_epochs_ran = variant_epochs = optimizer_steps = 0
    train_graph_forwards = validation_graph_forwards = validation_checks = 0
    prior_training_seconds = 0.0
    prior_peak_rss = 0
    prior_training_gpu: dict[str, int] = {}
    resume_state = load_latest_training_state(
        latest_state_path,
        resume=args.resume,
        run_config=run_config,
        model=model,
        optimizer=optimizer,
        early_stopper=early_stopper,
        rng=rng,
        device=device,
    )
    if resume_state:
        history = list(resume_state["history"])
        counters = resume_state["counters"]
        super_epochs_ran = int(resume_state["completed_super_epoch"])
        variant_epochs = int(counters["variant_epochs"])
        optimizer_steps = int(counters["optimizer_steps"])
        train_graph_forwards = int(counters["train_graph_forwards"])
        validation_graph_forwards = int(counters["validation_graph_forwards"])
        validation_checks = int(counters["validation_checks"])
        prior_training_seconds = float(resume_state["training_seconds_elapsed"])
        prior_peak_rss = int(resume_state.get("process_peak_rss_bytes", 0))
        prior_training_gpu = dict(resume_state.get("training_gpu", {}))

    reset_cuda_peak(device)
    training_start = time.perf_counter()
    for super_epoch in range(super_epochs_ran, super_epochs):
        if early_stopper.should_stop:
            break
        cycle_start = time.perf_counter()
        order = [variants[index] for index in rng.permutation(len(variants))]
        train_losses = {}
        for variant in order:
            shuffled = bundles[variant]["train_pos"][
                rng.permutation(len(bundles[variant]["train_pos"]))
            ]
            losses = []
            for start in range(0, len(shuffled), batch_sizes[variant]):
                positive_array = shuffled[start : start + batch_sizes[variant]]
                negative_array = sample_negative_triples(
                    positive_array, num_entities, neg_per_pos, rng
                )
                positives = torch.from_numpy(positive_array).long().to(device)
                negatives = torch.from_numpy(negative_array).long().to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                embeddings = model.encode(graphs[variant])
                train_graph_forwards += 1
                positive_scores = score_triples(model, embeddings, positives)
                negative_scores = score_triples(model, embeddings, negatives)
                # Preserve the original DHN WordNet objective: positive and
                # negative terms have equal weight regardless of neg_per_pos.
                loss = -(
                    F.logsigmoid(positive_scores).mean()
                    + F.logsigmoid(-negative_scores).mean()
                )
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))
                graphs[variant].x = None
                del embeddings, positive_scores, negative_scores, loss
            variant_epochs += 1
            train_losses[variant] = float(np.mean(losses))

        super_epochs_ran = super_epoch + 1
        do_eval = super_epochs_ran % eval_interval == 0
        validation_metrics: dict[str, dict[str, float]] = {}
        mean_val_mrr = float("nan")
        if do_eval:
            validation_checks += 1
            for variant in variants:
                validation_metrics[variant] = filtered_evaluate(
                    model,
                    graphs[variant],
                    _triples(shared["val_pos"]),
                    filters[variant],
                    device,
                    eval_batch_size,
                    binary_k,
                )
                validation_graph_forwards += 1
            mean_val_mrr = float(
                np.mean(
                    [
                        validation_metrics[variant]["filtered_MRR"]
                        for variant in variants
                    ]
                )
            )
            if early_stopper.update(mean_val_mrr):
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "metadata": {
                            "seed": seed,
                            "variants": variants,
                            "best_super_epoch": super_epochs_ran,
                            "optimizer_steps": optimizer_steps,
                            "selection_metric": "mean_filtered_MRR",
                            "model": "DHNWordNetLP",
                        },
                    },
                    checkpoint_path,
                )

        row: dict[str, Any] = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "updates_per_super_epoch": updates_per_super_epoch,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "validation_check": int(do_eval),
            "validation_checks_cumulative": validation_checks,
            "mean_val_filtered_MRR": mean_val_mrr,
            "best_mean_val_filtered_MRR": (
                early_stopper.best if np.isfinite(early_stopper.best) else None
            ),
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"batches_{variant}"] = batches_per_variant[variant]
            row[f"train_loss_{variant}"] = train_losses[variant]
            if do_eval:
                for metric, value in validation_metrics[variant].items():
                    row[f"val_{metric}_{variant}"] = value
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")
        elapsed = time.perf_counter() - training_start
        training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_state_path,
            run_config=run_config,
            model=model,
            optimizer=optimizer,
            early_stopper=early_stopper,
            rng=rng,
            completed_super_epoch=super_epochs_ran,
            counters={
                "variant_epochs": variant_epochs,
                "optimizer_steps": optimizer_steps,
                "train_graph_forwards": train_graph_forwards,
                "validation_graph_forwards": validation_graph_forwards,
                "validation_checks": validation_checks,
            },
            history=history,
            training_seconds_elapsed=prior_training_seconds + elapsed,
            process_peak_rss_bytes_value=peak_rss,
            training_gpu=training_gpu,
        )
        print(
            f"seed={seed} super_epoch={super_epochs_ran:04d} "
            f"steps={optimizer_steps} train_loss={row['mean_train_loss']:.6f} "
            f"val_filtered_MRR={mean_val_mrr:.6f} order={order}",
            flush=True,
        )
        if do_eval and early_stopper.should_stop:
            break

    training_seconds = prior_training_seconds + (
        time.perf_counter() - training_start
    )
    training_memory = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
    if np.isfinite(early_stopper.best):
        if not checkpoint_path.exists():
            raise RuntimeError("Best checkpoint is missing")
        model.load_state_dict(
            torch_load_full(checkpoint_path, map_location=device)["model"]
        )
        checkpoint_kind = "best_validation_filtered_MRR"
    else:
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "metadata": {
                    "seed": seed,
                    "variants": variants,
                    "checkpoint_kind": "final_weights_no_validation",
                },
            },
            checkpoint_path,
        )
        checkpoint_kind = "final_weights_no_validation"

    reset_cuda_peak(device)
    legacy_metrics: dict[str, dict[str, float]] = {}
    shared_metrics: dict[str, dict[str, float]] = {}
    shared_frames: dict[str, pd.DataFrame] = {}
    test_graph_forwards = 0
    for variant in variants:
        metrics = filtered_evaluate(
            model,
            graphs[variant],
            _triples(shared["test_pos"]),
            filters[variant],
            device,
            eval_batch_size,
            binary_k,
        )
        test_graph_forwards += 1
        metrics["edge_count"] = float(bundles[variant]["edge_count"])
        metrics["training_triples"] = float(len(bundles[variant]["train_pos"]))
        legacy_metrics[variant] = metrics
        candidate_metrics, frame = shared_candidate_frame(
            model,
            graphs[variant],
            shared["test_candidates"],
            device,
            args.score_batch_size,
        )
        test_graph_forwards += 1
        shared_metrics[variant] = candidate_metrics
        shared_frames[variant] = frame
        frame.to_csv(
            seed_dir / f"shared_candidate_test_scores_{variant}.csv", index=False
        )

    inference_memory = cuda_memory_stats(device)
    invariance = triple_link_invariance_rows(shared_frames, 0.5)
    pd.DataFrame(invariance).to_csv(
        seed_dir / "pairwise_invariance.csv", index=False
    )
    pd.DataFrame(
        [
            {"variant": variant, **metrics}
            for variant, metrics in legacy_metrics.items()
        ]
    ).to_csv(seed_dir / "legacy_test_metrics_by_variant.csv", index=False)
    pd.DataFrame(
        [
            {"variant": variant, **metrics}
            for variant, metrics in shared_metrics.items()
        ]
    ).to_csv(seed_dir / "shared_candidate_metrics_by_variant.csv", index=False)

    expected_steps = super_epochs_ran * updates_per_super_epoch
    expected_variant_epochs = super_epochs_ran * len(variants)
    if optimizer_steps != expected_steps or variant_epochs != expected_variant_epochs:
        raise AssertionError("Super-epoch/update accounting mismatch")
    summary = {
        "dataset": "WORDNET",
        "model": "DHNWordNetLP",
        "seed": seed,
        "variants": variants,
        "epoch_accounting": {
            "definition": (
                "one super-epoch visits every variant and every supervised "
                "triple batch in that variant"
            ),
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "batches_per_variant": batches_per_variant,
            "updates_per_super_epoch": updates_per_super_epoch,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
            "validation_checks": validation_checks,
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "training_seconds": training_seconds,
        "selection_metric": "mean_filtered_MRR",
        "training_objective": (
            "-mean(logsigmoid(positive))-mean(logsigmoid(-negative))"
        ),
        "splits_npz": str(args.splits_npz.resolve()),
        "split_protocol": "official_leakage_free_four_variant_wordnet_splits",
        "augmentation_protocol": "selected_variant_shared_model",
        "canonical_archive_variants": list(VARIANTS),
        "best_mean_val_filtered_MRR": (
            early_stopper.best if np.isfinite(early_stopper.best) else None
        ),
        "checkpoint_kind": checkpoint_kind,
        "mean_legacy_test_metrics": mean_dict(list(legacy_metrics.values())),
        "per_variant_legacy_test_metrics": legacy_metrics,
        "mean_shared_candidate_metrics": mean_dict(list(shared_metrics.values())),
        "per_variant_shared_candidate_metrics": shared_metrics,
        "pairwise_invariance": invariance,
        "memory": {
            **model_memory_bytes(model),
            "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
            "process_peak_rss_bytes": peak_rss,
            "training_gpu": training_memory,
            "inference_gpu": inference_memory,
        },
    }
    write_json(seed_dir / "summary.json", summary)
    return summary


def flatten_seed_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    accounting = summary["epoch_accounting"]
    memory = summary["memory"]
    row = {
        "seed": summary["seed"],
        "variants": ",".join(summary["variants"]),
        "super_epochs_ran": accounting["super_epochs_ran"],
        "variant_epochs_ran": accounting["variant_epochs_ran"],
        "updates_per_super_epoch": accounting["updates_per_super_epoch"],
        "optimizer_steps": accounting["optimizer_steps"],
        "training_seconds": summary["training_seconds"],
        "best_mean_val_filtered_MRR": summary["best_mean_val_filtered_MRR"],
        **{
            f"mean_legacy_test_{key}": value
            for key, value in summary["mean_legacy_test_metrics"].items()
        },
        **{
            f"mean_shared_candidate_{key}": value
            for key, value in summary["mean_shared_candidate_metrics"].items()
        },
        "parameter_bytes": memory["parameter_bytes"],
        "buffer_bytes": memory["buffer_bytes"],
        "static_model_bytes": memory["static_model_bytes"],
        "checkpoint_bytes": memory["checkpoint_bytes"],
        "process_peak_rss_bytes": memory["process_peak_rss_bytes"],
    }
    for phase in ("training_gpu", "inference_gpu"):
        prefix = phase.removesuffix("_gpu")
        for key, value in memory[phase].items():
            row[f"{prefix}_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint three-variant WordNet augmentation for DHN"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/wordnet_augmentation.yaml"))
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=(
            "Defaults to no_changes,all_inverse_edges,universal_edges; "
            "transitive_edges is an optional ablation"
        ),
    )
    parser.add_argument("--seeds", default="")
    parser.add_argument(
        "--splits-npz",
        type=Path,
        default=Path("data/preprocessed/WordNet_four_variant_splits.npz"),
    )
    parser.add_argument("--bundle-root", type=Path, default=Path("data/preprocessed"))
    parser.add_argument(
        "--bundle-pattern", default="WordNet3Hop_dhn_lp_{variant}.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dhn_augmentation/WORDNET"),
    )
    parser.add_argument(
        "--super-epochs", "--epochs", dest="super_epochs", type=int, default=None
    )
    parser.add_argument(
        "--eval-interval", "--eval_interval",
        dest="eval_interval", type=int, default=None,
    )
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--patience-evals", type=int, default=None)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None
    )
    parser.add_argument(
        "--neg-per-pos", "--neg_per_pos",
        dest="neg_per_pos", type=int, default=None,
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--binary-k", type=int, default=None)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--fixed-negatives", type=int, default=50)
    parser.add_argument("--candidate-seed", type=int, default=1566911444)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the shared split, bundles, and model contract, then exit",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.binary_k is not None:
        config = dict(config)
        config["eval"] = dict(config["eval"])
        config["eval"]["binary_k"] = args.binary_k
    variants = canonicalize_variants(args.variants)
    seeds = (
        [int(value) for value in parse_csv(args.seeds)]
        if args.seeds
        else [int(value) for value in config["seeds"]]
    )
    args.splits_npz = args.splits_npz.resolve()
    args.bundle_root = args.bundle_root.resolve()
    if args.preflight_only:
        bundles, shared = prepare_bundles(
            args.splits_npz,
            args.bundle_root,
            args.bundle_pattern,
            variants,
            args.fixed_negatives,
            args.candidate_seed,
        )
        validate_model_bundle_contract(config, bundles)
        print(
            f"[OK] WordNet augmentation preflight: variants={variants}, "
            f"entities={int(shared['num_entities'])}, "
            f"relations={int(shared['num_relations'])}, seeds={seeds}"
        )
        for variant in variants:
            data = bundles[variant]["data"]
            mapping_counts = {
                name: 0 if mapping is None else int(mapping.shape[0])
                for name, mapping in data.mapping_index_dict.items()
            }
            print(
                f"  {variant}: train_triples="
                f"{len(bundles[variant]['train_pos'])} "
                f"edges={bundles[variant]['edge_count']} "
                f"mappings={mapping_counts} path={bundles[variant]['_path']}"
            )
        return
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_seed(args, config, variants, seed, output_root) for seed in seeds
    ]
    pd.DataFrame([flatten_seed_summary(summary) for summary in summaries]).to_csv(
        output_root / "seed_summary.csv", index=False
    )
    write_json(output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
