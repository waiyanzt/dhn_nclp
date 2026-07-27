"""Joint IMDb graph-variant augmentation for DHN link prediction.

For one IMDb task, one DHN-LP model, optimizer, and checkpoint are shared
across every selected graph variant.  A super-epoch visits every variant once
in seeded random order.  The model, dot-product decoder, pairwise log-sigmoid
loss, fixed positive/negative tables, and validation-loss checkpoint criterion
are inherited from ``experiments.link_prediction.imdb``.

Examples:
    python -m experiments.link_prediction.imdb_augmentation --task md
    python -m experiments.link_prediction.imdb_augmentation --task mg
    python -m experiments.link_prediction.imdb_augmentation --task ml
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import kendalltau

from dhn.augmentation_utils import (
    EarlyStopper,
    atomic_torch_save,
    atomic_write_csv,
    checkpoint_size_bytes,
    cuda_memory_stats,
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
    write_json,
)
from dhn.utils import get_act_module, get_optimizer
from experiments.link_prediction.imdb import (
    DHN_LP,
    TASK_TARGET_TYPE,
    compute_metrics,
    pairwise_logsigmoid_loss,
    resolve_layers_config,
    score_neg_table,
    score_pairs,
)


VALID_VARIANTS = {
    "md": ("v1", "v3"),
    "mg": ("v1", "v2", "v3", "v4"),
    "ml": ("v1", "v2", "v3", "v4"),
}
TARGET_LOCAL_COLUMN = {
    "md": "director_local",
    "mg": "genre_id",
    "ml": "link_local",
}
SPLIT_NAMES = (
    "train_pos",
    "train_neg",
    "val_pos",
    "val_neg",
    "test_pos",
    "test_neg",
)


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Expected a nonempty comma-separated list without duplicates")
    return values


def parse_variants(task: str, value: str) -> list[str]:
    variants = [item.lower() for item in parse_csv(value)]
    invalid = [item for item in variants if item not in VALID_VARIANTS[task]]
    if invalid:
        raise ValueError(
            f"Invalid variants for {task}: {invalid}; valid variants are "
            f"{list(VALID_VARIANTS[task])}"
        )
    return variants


def parse_bundle_overrides(
    values: list[str], task: str, workspace_root: Path
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --bundle {value!r}; expected variant=/path/to/file.pt"
            )
        variant, raw_path = value.split("=", 1)
        variant = variant.strip().lower()
        if variant not in VALID_VARIANTS[task]:
            raise ValueError(f"Invalid bundle override variant {variant!r} for {task}")
        path = Path(raw_path).expanduser()
        overrides[variant] = path if path.is_absolute() else workspace_root / path
    return overrides


def default_bundle_path(task: str, variant: str) -> Path:
    return Path(f"data/preprocessed/IMDB_dhn_lp_{task}_{variant}.pt")


def torch_load_bundle(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assert_same_tensor(
    name: str, by_variant: Mapping[str, torch.Tensor]
) -> None:
    variants = list(by_variant)
    reference_variant = variants[0]
    reference = by_variant[reference_variant].cpu()
    for variant in variants[1:]:
        current = by_variant[variant].cpu()
        if (
            reference.shape != current.shape
            or reference.dtype != current.dtype
            or not torch.equal(reference, current)
        ):
            raise ValueError(
                f"{name} differs between {reference_variant} and {variant}: "
                f"{tuple(reference.shape)} vs {tuple(current.shape)}"
            )


def prepare_bundles(
    task: str,
    variants: list[str],
    workspace_root: Path,
    overrides: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    overrides = overrides or {}
    bundles: dict[str, dict[str, Any]] = {}
    missing: list[Path] = []
    for variant in variants:
        relative = overrides.get(variant, default_bundle_path(task, variant))
        path = relative if relative.is_absolute() else workspace_root / relative
        if not path.is_file():
            missing.append(path)
            continue
        bundle = torch_load_bundle(path)
        meta = bundle.get("meta", {})
        if meta.get("task") != task or meta.get("variant") != variant:
            raise ValueError(
                f"{path} identifies task/variant "
                f"{meta.get('task')!r}/{meta.get('variant')!r}, expected "
                f"{task!r}/{variant!r}"
            )
        bundle["_path"] = str(path.resolve())
        bundles[variant] = bundle
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Missing IMDb DHN link-prediction bundles:\n"
            f"{formatted}\n"
            "Generate them with:\n"
            "  python -m preprocess.imdb.link_prediction "
            f"--task {task} --variant {','.join(variants)}"
        )

    reference = bundles[variants[0]]
    reference_meta = reference["meta"]
    num_nodes = int(reference_meta["num_nodes_total"])
    num_nodes_per_type = dict(reference_meta["num_nodes_per_type"])
    patterns = tuple(reference_meta["patterns"])
    offsets = dict(reference["node_offsets"])
    for variant, bundle in bundles.items():
        data = bundle["data"]
        meta = bundle["meta"]
        if int(meta["num_nodes_total"]) != num_nodes or int(data.num_nodes) != num_nodes:
            raise ValueError(f"num_nodes differs for {variant}")
        if dict(meta["num_nodes_per_type"]) != num_nodes_per_type:
            raise ValueError(f"num_nodes_per_type differs for {variant}")
        if dict(bundle["node_offsets"]) != offsets:
            raise ValueError(f"node_offsets differs for {variant}")
        if tuple(meta["patterns"]) != patterns:
            raise ValueError(f"pattern set differs for {variant}")
        if set(data.mapping_index_dict) != set(patterns):
            raise ValueError(
                f"{variant} mapping keys {sorted(data.mapping_index_dict)} "
                f"do not match declared patterns {sorted(patterns)}"
            )

    for split_name in SPLIT_NAMES:
        assert_same_tensor(
            split_name,
            {
                variant: bundle["splits"][split_name]
                for variant, bundle in bundles.items()
            },
        )
    validate_split_ranges(reference, task)
    return bundles


def validate_split_ranges(bundle: Mapping[str, Any], task: str) -> None:
    offsets = bundle["node_offsets"]
    counts = bundle["meta"]["num_nodes_per_type"]
    movie_start = int(offsets["movie"])
    movie_stop = movie_start + int(counts["movie"])
    target_type = TASK_TARGET_TYPE[task]
    target_start = int(offsets[target_type])
    target_stop = target_start + int(counts[target_type])
    for split in ("train", "val", "test"):
        pos = bundle["splits"][f"{split}_pos"]
        neg = bundle["splits"][f"{split}_neg"]
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"{split}_pos must have shape (N, 2)")
        if neg.ndim != 2 or neg.shape[0] != pos.shape[0]:
            raise ValueError(f"{split}_neg must have shape (N, K)")
        if pos.numel() == 0 or neg.numel() == 0:
            raise ValueError(f"{split} positive/negative table must be nonempty")
        if not bool(((pos[:, 0] >= movie_start) & (pos[:, 0] < movie_stop)).all()):
            raise ValueError(f"{split}_pos contains an out-of-range movie ID")
        targets = torch.cat((pos[:, 1], neg.reshape(-1)))
        if not bool(((targets >= target_start) & (targets < target_stop)).all()):
            raise ValueError(f"{split} contains an out-of-range target ID")


def validate_model_bundle_contract(
    config: Mapping[str, Any], bundles: Mapping[str, Mapping[str, Any]]
) -> None:
    patterns = set(next(iter(bundles.values()))["meta"]["patterns"])
    configured_layers = config["model"]["layers_config"]
    if not configured_layers:
        raise ValueError("model.layers_config must contain at least one layer")
    for layer_number, layer in enumerate(configured_layers, start=1):
        if set(layer) != patterns:
            raise ValueError(
                f"Layer {layer_number} kernels {sorted(layer)} do not match "
                f"bundle patterns {sorted(patterns)}"
            )
    scheduler = config.get("training", {}).get("lr_scheduling", {})
    if scheduler.get("name"):
        raise ValueError(
            "Joint augmentation exact-resume currently requires "
            "training.lr_scheduling.name: null"
        )


def move_bundle_to_device(
    bundle: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    data = bundle["data"].to(device)
    data.mapping_index_dict = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in data.mapping_index_dict.items()
    }
    return {
        **bundle,
        "data": data,
        "splits": {
            name: tensor.to(device) for name, tensor in bundle["splits"].items()
        },
        "edge_count": int(data.edge_index.shape[1]),
        "mapping_counts": {
            name: 0 if value is None else int(value.shape[0])
            for name, value in data.mapping_index_dict.items()
        },
    }


def create_model(
    config: Mapping[str, Any], num_nodes: int
) -> DHN_LP:
    model_config = config["model"]
    in_dim = int(model_config["in_dim"])
    layers = resolve_layers_config(model_config["layers_config"], in_dim)
    activation = model_config["activation"]
    return DHN_LP(
        num_nodes=num_nodes,
        in_dim=in_dim,
        layers_config=layers,
        act_module=get_act_module(activation["name"]),
        **activation.get("kwargs", {}),
        **model_config.get("homconv_kwargs", {}),
    )


@torch.no_grad()
def evaluate_variant(
    model: DHN_LP,
    bundle: Mapping[str, Any],
    split: str,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    h = model.encode(bundle["data"])
    pos = bundle["splits"][f"{split}_pos"]
    neg = bundle["splits"][f"{split}_neg"]
    positive_scores = score_pairs(h, pos)
    negative_scores = score_neg_table(h, pos[:, 0], neg)
    loss = float(
        pairwise_logsigmoid_loss(positive_scores, negative_scores).cpu()
    )
    positive_cpu = positive_scores.detach().cpu()
    negative_cpu = negative_scores.detach().cpu()
    # DHN_LP.encode installs the learned embedding tensor as graph.x. It is
    # regenerated on every call, so do not retain one activation per variant.
    bundle["data"].x = None
    return loss, positive_cpu, negative_cpu


def iter_training_batches(
    pos: torch.Tensor,
    neg: torch.Tensor,
    batch_size: int,
    rng: np.random.RandomState,
):
    row_count = int(pos.shape[0])
    if batch_size <= 0:
        # Preserve the standalone DHN operation order for the default path.
        yield pos, neg
        return
    effective_batch_size = min(batch_size, row_count)
    order = rng.permutation(row_count)
    for start in range(0, row_count, effective_batch_size):
        indices = torch.as_tensor(
            order[start : start + effective_batch_size],
            dtype=torch.long,
            device=pos.device,
        )
        yield pos[indices], neg[indices]


def build_score_frame(
    task: str,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    offsets: Mapping[str, int],
) -> pd.DataFrame:
    pos = test_pos.detach().cpu().numpy()
    neg = test_neg.detach().cpu().numpy()
    raw_pos = positive_scores.detach().cpu().numpy()
    raw_neg = negative_scores.detach().cpu().numpy()
    probabilities_pos = torch.sigmoid(positive_scores.detach().cpu()).numpy()
    probabilities_neg = torch.sigmoid(negative_scores.detach().cpu()).numpy()
    ranks = (raw_neg >= raw_pos[:, None]).sum(axis=1) + 1
    target_offset = int(offsets[TASK_TARGET_TYPE[task]])
    movie_offset = int(offsets["movie"])
    target_column = TARGET_LOCAL_COLUMN[task]

    rows: list[dict[str, Any]] = []
    candidates_per_query = neg.shape[1] + 1
    for query_row in range(pos.shape[0]):
        base = {
            "query_row": query_row,
            "movie_local": int(pos[query_row, 0] - movie_offset),
            "rank_of_positive": int(ranks[query_row]),
        }
        rows.append(
            {
                **base,
                "candidate_position": 0,
                "candidate_id": query_row * candidates_per_query,
                target_column: int(pos[query_row, 1] - target_offset),
                "raw_score": float(raw_pos[query_row]),
                "score": float(probabilities_pos[query_row]),
                "label": 1,
            }
        )
        for negative_index in range(neg.shape[1]):
            candidate = int(neg[query_row, negative_index] - target_offset)
            rows.append(
                {
                    **base,
                    "candidate_position": negative_index + 1,
                    "candidate_id": (
                        query_row * candidates_per_query + negative_index + 1
                    ),
                    target_column: candidate,
                    "raw_score": float(raw_neg[query_row, negative_index]),
                    "score": float(probabilities_neg[query_row, negative_index]),
                    "label": 0,
                }
            )
    return pd.DataFrame(rows)


def _safe_tau(left: np.ndarray, right: np.ndarray) -> float:
    value = kendalltau(left, right, nan_policy="omit").statistic
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def link_invariance_rows(
    outputs: Mapping[str, pd.DataFrame], threshold: float
) -> list[dict[str, Any]]:
    variants = list(outputs)
    rows: list[dict[str, Any]] = []
    identity_columns = [
        "query_row",
        "candidate_position",
        "candidate_id",
        "movie_local",
        "label",
    ]
    for index, variant_a in enumerate(variants):
        for variant_b in variants[index + 1 :]:
            left = outputs[variant_a].sort_values(
                ["query_row", "candidate_position"]
            ).reset_index(drop=True)
            right = outputs[variant_b].sort_values(
                ["query_row", "candidate_position"]
            ).reset_index(drop=True)
            if len(left) != len(right) or not left[identity_columns].equals(
                right[identity_columns]
            ):
                raise ValueError(
                    f"Candidate tables do not align for {variant_a} and {variant_b}"
                )
            score_a = left["score"].to_numpy()
            score_b = right["score"].to_numpy()
            positive_a = left[left["label"] == 1].sort_values("query_row")
            positive_b = right[right["label"] == 1].sort_values("query_row")
            ranks_a = positive_a["rank_of_positive"].to_numpy()
            ranks_b = positive_b["rank_of_positive"].to_numpy()
            difference = score_a - score_b
            rows.append(
                {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "candidate_count": int(len(score_a)),
                    "query_count": int(len(ranks_a)),
                    "kendall_tau_scores": _safe_tau(score_a, score_b),
                    "prediction_agreement": float(
                        np.mean((score_a > threshold) == (score_b > threshold))
                    ),
                    "rank_agreement": float(np.mean(ranks_a == ranks_b)),
                    "kendall_tau_positive_ranks": _safe_tau(ranks_a, ranks_b),
                    "hits_at_1_agreement": float(
                        np.mean((ranks_a <= 1) == (ranks_b <= 1))
                    ),
                    "hits_at_3_agreement": float(
                        np.mean((ranks_a <= 3) == (ranks_b <= 3))
                    ),
                    "max_abs_score_diff": float(np.max(np.abs(difference))),
                    "mean_abs_score_diff": float(np.mean(np.abs(difference))),
                }
            )
    return rows


def rgcn_metric_names(metrics: Mapping[str, float], query_count: int) -> dict[str, float]:
    """Expose the existing DHN calculations under the RGCN report schema."""
    names = {
        "auc": "AUC",
        "ap": "AP",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "accuracy": "Accuracy",
        "mrr": "MRR",
    }
    output: dict[str, float] = {}
    for key, value in metrics.items():
        output_key = (
            f"Hits@{key.split('@', 1)[1]}"
            if key.startswith("hits@")
            else names.get(key, key)
        )
        output[output_key] = value
    output["ranking_queries"] = float(query_count)
    return output


def run_seed(
    args,
    config: Mapping[str, Any],
    variants: list[str],
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    set_determinism(seed)
    device = resolve_device(args.device or config.get("device", "auto"))
    cpu_bundles = prepare_bundles(
        args.task,
        variants,
        args.workspace_root,
        args.bundle_overrides,
    )
    validate_model_bundle_contract(config, cpu_bundles)
    bundles = {
        variant: move_bundle_to_device(bundle, device)
        for variant, bundle in cpu_bundles.items()
    }
    reference = bundles[variants[0]]
    model = create_model(config, int(reference["meta"]["num_nodes_total"])).to(device)

    training_config = config["training"]
    optimizer_config = training_config["optimizer"]
    optimizer_kwargs = dict(optimizer_config.get("kwargs", {}))
    if args.lr is not None:
        optimizer_kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        optimizer_kwargs["weight_decay"] = args.weight_decay
    optimizer = get_optimizer(optimizer_config["name"])(
        model.parameters(), **optimizer_kwargs
    )
    super_epochs = (
        args.super_epochs
        if args.super_epochs is not None
        else int(training_config.get("super_epochs", training_config["epochs"]))
    )
    patience = (
        args.patience
        if args.patience is not None
        else int(training_config.get("patience", 15))
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training_config.get("batch_size", 0))
    )
    grad_clip = (
        args.grad_clip
        if args.grad_clip is not None
        else float(training_config.get("grad_clip", 0.0))
    )
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(config.get("eval", {}).get("threshold", 0.5))
    )
    hits_k = tuple(int(value) for value in config.get("eval", {}).get("hits_k", (1, 3, 5)))
    if super_epochs < 1:
        raise ValueError("super_epochs must be >= 1")
    if batch_size < 0:
        raise ValueError("batch_size must be >= 0 (0 means full split)")
    if grad_clip < 0:
        raise ValueError("grad_clip must be >= 0")

    train_rows = int(reference["splits"]["train_pos"].shape[0])
    effective_batch_size = train_rows if batch_size == 0 else min(batch_size, train_rows)
    batches_per_variant = math.ceil(train_rows / effective_batch_size)
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    early_stopper = EarlyStopper("min", patience, min_delta=0.0)
    rng = np.random.RandomState(seed)
    run_config = {
        "dataset": "IMDB_LP",
        "model": "DHN",
        "task": args.task,
        "seed": seed,
        "variants": variants,
        "bundle_paths": {
            variant: cpu_bundles[variant]["_path"] for variant in variants
        },
        "model_config": config["model"],
        "optimizer": optimizer_config["name"],
        "optimizer_kwargs": optimizer_kwargs,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "grad_clip": grad_clip,
        "patience": patience,
        "early_stopping_min_delta": 0.0,
        "selection_metric": "mean_validation_pairwise_logsigmoid_loss",
        "threshold": threshold,
        "threshold_comparison": "strictly_greater_than",
        "hits_k": hits_k,
    }

    history: list[dict[str, Any]] = []
    super_epochs_ran = variant_epochs = optimizer_steps = 0
    train_graph_forwards = validation_graph_forwards = 0
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
    if resume_state is not None:
        history = list(resume_state["history"])
        counters = resume_state["counters"]
        super_epochs_ran = int(resume_state["completed_super_epoch"])
        variant_epochs = int(counters["variant_epochs"])
        optimizer_steps = int(counters["optimizer_steps"])
        train_graph_forwards = int(counters["train_graph_forwards"])
        validation_graph_forwards = int(counters["validation_graph_forwards"])
        prior_training_seconds = float(resume_state["training_seconds_elapsed"])
        prior_peak_rss = int(resume_state.get("process_peak_rss_bytes", 0))
        prior_training_gpu = dict(resume_state.get("training_gpu", {}))

    reset_cuda_peak(device)
    training_start = time.perf_counter()
    for super_epoch in range(super_epochs_ran, super_epochs):
        if early_stopper.should_stop:
            print("[resume] Early stopping already reached; skipping training.")
            break
        cycle_start = time.perf_counter()
        variant_order = [
            variants[index] for index in rng.permutation(len(variants))
        ]
        train_losses: dict[str, float] = {}
        for variant in variant_order:
            bundle = bundles[variant]
            batch_losses: list[float] = []
            train_pos = bundle["splits"]["train_pos"]
            train_neg = bundle["splits"]["train_neg"]
            for pos_batch, neg_batch in iter_training_batches(
                train_pos, train_neg, batch_size, rng
            ):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                embeddings = model.encode(bundle["data"])
                train_graph_forwards += 1
                positive_scores = score_pairs(embeddings, pos_batch)
                negative_scores = score_neg_table(
                    embeddings, pos_batch[:, 0], neg_batch
                )
                loss = pairwise_logsigmoid_loss(positive_scores, negative_scores)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer_steps += 1
                batch_losses.append(float(loss.detach().cpu()))
                bundle["data"].x = None
                del embeddings, positive_scores, negative_scores, loss
            variant_epochs += 1
            train_losses[variant] = float(np.mean(batch_losses))

        validation_losses: dict[str, float] = {}
        for variant in variants:
            validation_losses[variant], _, _ = evaluate_variant(
                model, bundles[variant], "val"
            )
            validation_graph_forwards += 1
        mean_val_loss = float(np.mean(list(validation_losses.values())))
        super_epochs_ran = super_epoch + 1
        if early_stopper.update(mean_val_loss):
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "metadata": {
                        "task": args.task,
                        "seed": seed,
                        "variants": variants,
                        "best_super_epoch": super_epochs_ran,
                        "optimizer_steps": optimizer_steps,
                        "selection_metric": (
                            "mean_validation_pairwise_logsigmoid_loss"
                        ),
                    },
                },
                checkpoint_path,
            )

        row: dict[str, Any] = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(variant_order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "train_graph_forwards_cumulative": train_graph_forwards,
            "validation_graph_forwards_cumulative": validation_graph_forwards,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_val_loss,
            "best_mean_val_loss": early_stopper.best,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"train_loss_{variant}"] = train_losses[variant]
            row[f"val_loss_{variant}"] = validation_losses[variant]
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")

        elapsed_segment = time.perf_counter() - training_start
        training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_state_path,
            dataset="IMDB_LP",
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
            },
            history=history,
            training_seconds_elapsed=prior_training_seconds + elapsed_segment,
            process_peak_rss_bytes_value=peak_rss,
            training_gpu=training_gpu,
        )
        print(
            f"task={args.task} seed={seed} super_epoch={super_epochs_ran:03d} "
            f"variant_epochs={variant_epochs} optimizer_steps={optimizer_steps} "
            f"mean_train_loss={row['mean_train_loss']:.6f} "
            f"mean_val_loss={mean_val_loss:.6f} "
            f"best={early_stopper.best:.6f} order={variant_order}",
            flush=True,
        )
        if early_stopper.should_stop:
            print("Early stopping after a balanced super-epoch.", flush=True)
            break

    training_seconds = prior_training_seconds + (
        time.perf_counter() - training_start
    )
    training_memory = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
    if not checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    model.load_state_dict(
        torch_load_full(checkpoint_path, map_location=device)["model"]
    )

    reset_cuda_peak(device)
    per_variant_metrics: dict[str, dict[str, float]] = {}
    outputs: dict[str, pd.DataFrame] = {}
    test_graph_forwards = 0
    for variant in variants:
        _, positive_scores, negative_scores = evaluate_variant(
            model, bundles[variant], "test"
        )
        test_graph_forwards += 1
        metrics = rgcn_metric_names(
            compute_metrics(
                positive_scores,
                negative_scores,
                hits_k=hits_k,
                threshold=threshold,
            ),
            query_count=int(positive_scores.shape[0]),
        )
        metrics["edge_count"] = float(bundles[variant]["edge_count"])
        for pattern, count in bundles[variant]["mapping_counts"].items():
            metrics[f"{pattern}_mapping_count"] = float(count)
        per_variant_metrics[variant] = metrics
        frame = build_score_frame(
            args.task,
            reference["splits"]["test_pos"],
            reference["splits"]["test_neg"],
            positive_scores,
            negative_scores,
            reference["node_offsets"],
        )
        outputs[variant] = frame
        frame.to_csv(seed_dir / f"test_scores_{variant}.csv", index=False)
        frame.to_csv(
            seed_dir
            / f"IMDB_dhn_lp_augmentation_{args.task}_{variant}_seed{seed}_scores.csv",
            index=False,
        )

    inference_memory = cuda_memory_stats(device)
    invariance = link_invariance_rows(outputs, threshold)
    pd.DataFrame(invariance).to_csv(
        seed_dir / "pairwise_invariance.csv", index=False
    )
    pd.DataFrame(
        [
            {"task": args.task, "variant": variant, **metrics}
            for variant, metrics in per_variant_metrics.items()
        ]
    ).to_csv(seed_dir / "test_metrics_by_variant.csv", index=False)

    expected_variant_epochs = super_epochs_ran * len(variants)
    expected_steps = expected_variant_epochs * batches_per_variant
    if variant_epochs != expected_variant_epochs or optimizer_steps != expected_steps:
        raise AssertionError(
            f"Epoch accounting mismatch: variant epochs "
            f"{variant_epochs}/{expected_variant_epochs}, optimizer steps "
            f"{optimizer_steps}/{expected_steps}"
        )
    summary = {
        "dataset": "IMDB_LP",
        "model": "DHN",
        "task": args.task,
        "seed": seed,
        "variants": variants,
        "shared_split_shapes": {
            name: list(reference["splits"][name].shape) for name in SPLIT_NAMES
        },
        "negative_candidates_per_positive": {
            split: int(reference["splits"][f"{split}_neg"].shape[1])
            for split in ("train", "val", "test")
        },
        "epoch_accounting": {
            "definition": (
                "one super-epoch visits every selected graph variant once; "
                "within a variant, one optimizer update is made per positive-row "
                "batch"
            ),
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "batches_per_variant": batches_per_variant,
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "training_seconds": training_seconds,
        "selection_metric": "mean_validation_pairwise_logsigmoid_loss",
        "best_mean_val_loss": early_stopper.best,
        "mean_test_metrics": mean_dict(list(per_variant_metrics.values())),
        "per_variant_test_metrics": per_variant_metrics,
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
        "task": summary["task"],
        "seed": summary["seed"],
        "variants": ",".join(summary["variants"]),
        "batch_size": accounting["batch_size"],
        "effective_batch_size": accounting["effective_batch_size"],
        "batches_per_variant": accounting["batches_per_variant"],
        "super_epochs_ran": accounting["super_epochs_ran"],
        "variant_epochs_ran": accounting["variant_epochs_ran"],
        "optimizer_steps": accounting["optimizer_steps"],
        "training_seconds": summary["training_seconds"],
        "best_mean_val_loss": summary["best_mean_val_loss"],
        **{
            f"mean_test_{key}": value
            for key, value in summary["mean_test_metrics"].items()
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
        description="Joint IMDb graph-variant augmentation for DHN LP"
    )
    parser.add_argument("--task", required=True, choices=tuple(VALID_VARIANTS))
    parser.add_argument("--config", type=Path, default=Path("configs/imdb_lp.yaml"))
    parser.add_argument(
        "--variants",
        default="",
        help="Default: every valid baseline variant for the selected task",
    )
    parser.add_argument("--seeds", default="")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("."),
        help="Prefix for default relative bundle paths",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Override a bundle path, e.g. --bundle v1=/path/bundle.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dhn_augmentation/IMDB_LP"),
    )
    parser.add_argument(
        "--super-epochs",
        "--epochs",
        dest="super_epochs",
        type=int,
        default=None,
    )
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Positive rows per optimizer update; 0 preserves full-batch DHN",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all bundles/configuration and exit before model training",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    args.workspace_root = args.workspace_root.resolve()
    variants = (
        parse_variants(args.task, args.variants)
        if args.variants
        else list(VALID_VARIANTS[args.task])
    )
    args.bundle_overrides = parse_bundle_overrides(
        args.bundle, args.task, args.workspace_root
    )
    seeds = (
        [int(value) for value in parse_csv(args.seeds)]
        if args.seeds
        else [int(value) for value in config.get(
            "seeds", (1566911444, 20241017, 20251017)
        )]
    )

    if args.preflight_only:
        bundles = prepare_bundles(
            args.task,
            variants,
            args.workspace_root,
            args.bundle_overrides,
        )
        validate_model_bundle_contract(config, bundles)
        reference = bundles[variants[0]]
        print(
            f"[OK] IMDb LP joint augmentation preflight: task={args.task}, "
            f"variants={variants}, nodes={reference['meta']['num_nodes_total']}, "
            f"train_pos={tuple(reference['splits']['train_pos'].shape)}, "
            f"seeds={seeds}"
        )
        for variant in variants:
            data = bundles[variant]["data"]
            mapping_counts = {
                name: 0 if mapping is None else int(mapping.shape[0])
                for name, mapping in data.mapping_index_dict.items()
            }
            print(
                f"  {variant}: edges={int(data.edge_index.shape[1])} "
                f"mappings={mapping_counts} path={bundles[variant]['_path']}"
            )
        return

    task_output_root = (args.output_dir / args.task).resolve()
    task_output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_seed(args, config, variants, seed, task_output_root)
        for seed in seeds
    ]
    pd.DataFrame([flatten_seed_summary(summary) for summary in summaries]).to_csv(
        task_output_root / "seed_summary.csv", index=False
    )
    write_json(task_output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"[OK] Results written under {task_output_root}")


if __name__ == "__main__":
    main()
