"""Joint Freebase graph-variant augmentation for DHN node classification.

One featureless DHN, optimizer, and checkpoint are shared across the
``unchanged`` and ``exact_2`` Freebase graphs.  A balanced super-epoch visits
each selected variant exactly once in seeded random order.  The best shared
checkpoint minimizes mean validation cross-entropy across both variants.

The preprocessed Freebase bundles deliberately contain only ``p1`` and ``c2``
mappings rooted at labeled BOOK nodes.  This runner therefore requires the
one-layer DHN configuration for which those rooted mappings are exact at every
supervised node.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from dhn.augmentation_utils import (
    EarlyStopper,
    atomic_torch_save,
    atomic_write_csv,
    checkpoint_size_bytes,
    classification_invariance_rows,
    classification_metrics,
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
from dhn.utils import get_act_module, get_criterion, get_optimizer
from experiments.node_classification.train import (
    FeaturelessDHNNodeClassifier,
    resolve_layers_config,
)


BUNDLE_PATHS = {
    "unchanged": Path("data/preprocessed/Freebase_dhn_nc_unchanged.pt"),
    "exact_2": Path("data/preprocessed/Freebase_dhn_nc_exact_2.pt"),
}
DEFAULT_VARIANTS = ("unchanged", "exact_2")
VARIANT_ALIASES = {
    "no_changes": "unchanged",
    "no changes": "unchanged",
    "exact2": "exact_2",
    "exact 2": "exact_2",
    "exact 2-hop": "exact_2",
}


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(
            "Expected a nonempty comma-separated list without duplicates"
        )
    return values


def canonical_variant(value: str) -> str:
    lowered = value.strip().lower()
    return VARIANT_ALIASES.get(lowered, lowered)


def parse_variants(value: str) -> list[str]:
    variants = [canonical_variant(item) for item in parse_csv(value)]
    unknown = sorted(set(variants) - set(BUNDLE_PATHS))
    if unknown:
        raise ValueError(f"Unknown Freebase variants: {unknown}")
    if len(variants) != len(set(variants)):
        raise ValueError("Variant aliases collapse to duplicate variants")
    return variants


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
        if reference.shape != current.shape or not torch.equal(reference, current):
            raise ValueError(
                f"{name} differs between {reference_variant} and {variant}: "
                f"{tuple(reference.shape)} vs {tuple(current.shape)}"
            )


def prepare_bundles(
    variants: list[str],
    workspace_root: Path,
    bundle_overrides: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    overrides = bundle_overrides or {}
    bundles: dict[str, dict[str, Any]] = {}
    missing = []
    for variant in variants:
        relative = overrides.get(variant, BUNDLE_PATHS[variant])
        path = relative if relative.is_absolute() else workspace_root / relative
        if not path.is_file():
            missing.append(path)
            continue
        bundle = torch_load_bundle(path)
        declared_variant = canonical_variant(
            str(bundle.get("meta", {}).get("variant", ""))
        )
        if declared_variant != variant:
            raise ValueError(
                f"{path} identifies variant {declared_variant!r}, "
                f"expected {variant!r}"
            )
        bundle["_path"] = str(path.resolve())
        bundles[variant] = bundle

    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Missing Freebase DHN bundles:\n"
            f"{formatted}\n"
            "Generate them with:\n"
            "  python -m preprocess.freebase.node_classification "
            "--variants unchanged exact_2 "
            "--raw-root data/raw/dataset_variant_3hops_filter "
            "--out-dir data/preprocessed"
        )

    reference = bundles[variants[0]]
    num_features = int(reference["num_features"])
    num_classes = int(reference["num_classes"])
    num_nodes = int(reference["data"].num_nodes)
    patterns = tuple(reference["meta"].get("patterns", ()))
    for variant, bundle in bundles.items():
        data = bundle["data"]
        if data.x is not None:
            raise ValueError(
                f"{variant} contains input features; Freebase DHN expects "
                "shared learned node embeddings"
            )
        if int(bundle["num_features"]) != num_features:
            raise ValueError(f"num_features differs for {variant}")
        if int(bundle["num_classes"]) != num_classes:
            raise ValueError(f"num_classes differs for {variant}")
        if int(data.num_nodes) != num_nodes:
            raise ValueError(f"num_nodes differs for {variant}")
        if tuple(bundle["meta"].get("patterns", ())) != patterns:
            raise ValueError(f"pattern set differs for {variant}")
        if set(data.mapping_index_dict) != set(patterns):
            raise ValueError(
                f"{variant} mapping keys {sorted(data.mapping_index_dict)} "
                f"do not match declared patterns {sorted(patterns)}"
            )
        if bundle["meta"].get("mapping_scope") != "labeled_target_roots":
            raise ValueError(
                f"{variant} does not declare labeled-target-root mappings"
            )

    assert_same_tensor(
        "labels", {variant: bundle["data"].y for variant, bundle in bundles.items()}
    )
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert_same_tensor(
            mask_name,
            {
                variant: getattr(bundle["data"], mask_name)
                for variant, bundle in bundles.items()
            },
        )
    assert_same_tensor(
        "p1 mappings",
        {
            variant: bundle["data"].mapping_index_dict["p1"]
            for variant, bundle in bundles.items()
        },
    )
    return bundles


def validate_model_bundle_contract(
    config: Mapping[str, Any], bundles: Mapping[str, Mapping[str, Any]]
) -> None:
    model_config = config["model"]
    if not model_config.get("learned_node_embeddings", False):
        raise ValueError("Freebase augmentation requires learned_node_embeddings=true")
    if model_config.get("agg") is not None:
        raise ValueError("Freebase node classification requires model.agg: null")

    configured_layers = model_config.get("layers_config", [])
    if len(configured_layers) != 1:
        raise ValueError(
            "Target-rooted Freebase mappings are exact only for the configured "
            "one-layer DHN; expected exactly one model layer"
        )
    patterns = set(next(iter(bundles.values()))["meta"]["patterns"])
    kernels = set(configured_layers[0])
    if kernels != patterns:
        raise ValueError(
            f"Layer kernels {sorted(kernels)} do not match bundle patterns "
            f"{sorted(patterns)}"
        )
    if patterns != {"p1", "c2"}:
        raise ValueError(
            f"Expected the tractable Freebase p1,c2 contract, got {sorted(patterns)}"
        )
    scheduler = config.get("training", {}).get("lr_scheduling", {})
    if scheduler.get("name"):
        raise ValueError(
            "Exact resume requires training.lr_scheduling.name: null"
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
        "train_idx": data.train_mask.nonzero(as_tuple=False).squeeze(1),
        "val_idx": data.val_mask.nonzero(as_tuple=False).squeeze(1),
        "test_idx": data.test_mask.nonzero(as_tuple=False).squeeze(1),
        "mapping_counts": {
            name: 0 if value is None else int(value.shape[0])
            for name, value in data.mapping_index_dict.items()
        },
    }


@torch.no_grad()
def evaluate_variant(
    model: torch.nn.Module,
    bundle: Mapping[str, Any],
    split: str,
    criterion,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    logits = model(bundle["data"])
    indices = bundle[f"{split}_idx"]
    loss = float(criterion(logits[indices], bundle["data"].y[indices]).cpu())
    metrics = classification_metrics(logits, bundle["data"].y, indices)
    return loss, metrics, logits.detach().cpu(), indices.detach().cpu()


def create_model(
    config: Mapping[str, Any], num_nodes: int, num_features: int, num_classes: int
) -> FeaturelessDHNNodeClassifier:
    model_config = config["model"]
    layers = resolve_layers_config(model_config["layers_config"], num_features)
    activation = model_config["activation"]
    return FeaturelessDHNNodeClassifier(
        num_nodes=num_nodes,
        embedding_dim=num_features,
        out_dim=num_classes,
        layers_config=layers,
        act_module=get_act_module(activation["name"]),
        agg=model_config.get("agg"),
        **activation.get("kwargs", {}),
        **model_config.get("homconv_kwargs", {}),
    )


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
        variants, args.workspace_root, args.bundle_overrides
    )
    validate_model_bundle_contract(config, cpu_bundles)
    bundles = {
        variant: move_bundle_to_device(bundle, device)
        for variant, bundle in cpu_bundles.items()
    }
    reference = bundles[variants[0]]
    model = create_model(
        config,
        int(reference["data"].num_nodes),
        int(reference["num_features"]),
        int(reference["num_classes"]),
    ).to(device)

    training_config = config["training"]
    augmentation_config = config.get("augmentation", {})
    loss_config = training_config["loss"]
    criterion = get_criterion(loss_config["name"])(
        **loss_config.get("kwargs", {})
    )
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
        else int(augmentation_config.get("super_epochs", training_config["epochs"]))
    )
    patience = (
        args.patience
        if args.patience is not None
        else int(augmentation_config.get("patience", 30))
    )
    grad_clip = (
        args.grad_clip
        if args.grad_clip is not None
        else float(augmentation_config.get("grad_clip", 0.0))
    )
    if super_epochs < 1:
        raise ValueError("super_epochs must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if grad_clip < 0:
        raise ValueError("grad_clip must be >= 0")

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    early_stopper = EarlyStopper("min", patience, min_delta=1e-6)
    rng = np.random.RandomState(seed)
    run_config = {
        "dataset": "FREEBASE",
        "model": "DHN",
        "seed": seed,
        "variants": variants,
        "bundle_paths": {
            variant: cpu_bundles[variant]["_path"] for variant in variants
        },
        "model_config": config["model"],
        "loss": loss_config,
        "optimizer": optimizer_config["name"],
        "optimizer_kwargs": optimizer_kwargs,
        "grad_clip": grad_clip,
        "patience": patience,
        "early_stopping_min_delta": 1e-6,
        "selection_metric": "mean_validation_cross_entropy",
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
        order = [variants[index] for index in rng.permutation(len(variants))]
        train_losses: dict[str, float] = {}
        for variant in order:
            bundle = bundles[variant]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(bundle["data"])
            train_graph_forwards += 1
            loss = criterion(
                logits[bundle["train_idx"]],
                bundle["data"].y[bundle["train_idx"]],
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer_steps += 1
            variant_epochs += 1
            train_losses[variant] = float(loss.detach().cpu())
            del logits, loss

        validation_losses = {}
        validation_metrics = {}
        for variant in variants:
            loss_value, metrics, _, _ = evaluate_variant(
                model, bundles[variant], "val", criterion
            )
            validation_graph_forwards += 1
            validation_losses[variant] = loss_value
            validation_metrics[variant] = metrics
        mean_val_loss = float(np.mean(list(validation_losses.values())))
        mean_val_macro_f1 = float(
            np.mean(
                [validation_metrics[variant]["Macro_F1"] for variant in variants]
            )
        )
        super_epochs_ran = super_epoch + 1
        if early_stopper.update(mean_val_loss):
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "metadata": {
                        "seed": seed,
                        "variants": variants,
                        "best_super_epoch": super_epochs_ran,
                        "optimizer_steps": optimizer_steps,
                        "selection_metric": "mean_validation_cross_entropy",
                    },
                },
                checkpoint_path,
            )

        row: dict[str, Any] = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "train_graph_forwards_cumulative": train_graph_forwards,
            "validation_graph_forwards_cumulative": validation_graph_forwards,
            "updates_per_super_epoch": len(variants),
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_val_loss,
            "best_mean_val_loss": early_stopper.best,
            "mean_val_macro_f1": mean_val_macro_f1,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"train_loss_{variant}"] = train_losses[variant]
            row[f"val_loss_{variant}"] = validation_losses[variant]
            row[f"val_macro_f1_{variant}"] = validation_metrics[variant][
                "Macro_F1"
            ]
            row[f"val_accuracy_{variant}"] = validation_metrics[variant][
                "Accuracy"
            ]
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")

        segment_seconds = time.perf_counter() - training_start
        training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_state_path,
            dataset="FREEBASE",
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
            training_seconds_elapsed=prior_training_seconds + segment_seconds,
            process_peak_rss_bytes_value=peak_rss,
            training_gpu=training_gpu,
        )
        print(
            f"seed={seed} super_epoch={super_epochs_ran:03d} "
            f"variant_epochs={variant_epochs} optimizer_steps={optimizer_steps} "
            f"mean_train_loss={row['mean_train_loss']:.6f} "
            f"mean_val_loss={mean_val_loss:.6f} "
            f"mean_val_macro_f1={mean_val_macro_f1:.6f} "
            f"best={early_stopper.best:.6f} order={order}",
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
    outputs: dict[str, dict[str, np.ndarray]] = {}
    test_graph_forwards = 0
    for variant in variants:
        test_loss, metrics, logits, test_idx = evaluate_variant(
            model, bundles[variant], "test", criterion
        )
        test_graph_forwards += 1
        selected_logits = logits[test_idx]
        probabilities = torch.softmax(selected_logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        item_ids = test_idx.numpy().astype(np.int64)
        labels = (
            bundles[variant]["data"].y[test_idx.to(device)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        frame = pd.DataFrame(
            {
                "node_id": item_ids,
                "label": labels,
                "prediction": predictions,
                "confidence": confidence,
            }
        )
        for class_id in range(probabilities.shape[1]):
            frame[f"prob_class_{class_id}"] = probabilities[:, class_id]
            frame[f"logit_class_{class_id}"] = selected_logits[
                :, class_id
            ].numpy()
        frame.to_csv(seed_dir / f"test_scores_{variant}.csv", index=False)

        metrics["Cross_Entropy"] = test_loss
        for pattern, count in bundles[variant]["mapping_counts"].items():
            metrics[f"{pattern}_mapping_count"] = float(count)
        per_variant_metrics[variant] = metrics
        outputs[variant] = {
            "item_id": item_ids,
            "logits": selected_logits.numpy(),
            "probabilities": probabilities,
            "prediction": predictions,
            "confidence": confidence,
        }

    inference_memory = cuda_memory_stats(device)
    invariance = classification_invariance_rows(outputs)
    pd.DataFrame(invariance).to_csv(
        seed_dir / "pairwise_invariance.csv", index=False
    )
    pd.DataFrame(
        [
            {"variant": variant, **metrics}
            for variant, metrics in per_variant_metrics.items()
        ]
    ).to_csv(seed_dir / "test_metrics_by_variant.csv", index=False)

    expected_steps = super_epochs_ran * len(variants)
    if optimizer_steps != expected_steps or variant_epochs != expected_steps:
        raise AssertionError(
            f"Epoch accounting mismatch: optimizer steps "
            f"{optimizer_steps}/{expected_steps}, variant epochs "
            f"{variant_epochs}/{expected_steps}"
        )
    summary = {
        "dataset": "FREEBASE",
        "model": "DHN",
        "seed": seed,
        "variants": variants,
        "epoch_accounting": {
            "definition": (
                "one super-epoch is one full labeled-node optimizer update "
                "on every selected graph variant"
            ),
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "updates_per_super_epoch": len(variants),
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "training_seconds": training_seconds,
        "selection_metric": "mean_validation_cross_entropy",
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
        "seed": summary["seed"],
        "variants": ",".join(summary["variants"]),
        "super_epochs_ran": accounting["super_epochs_ran"],
        "variant_epochs_ran": accounting["variant_epochs_ran"],
        "updates_per_super_epoch": accounting["updates_per_super_epoch"],
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


def parse_bundle_overrides(values: list[str]) -> dict[str, Path]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --bundle {value!r}; expected variant=/path/to/file.pt"
            )
        variant, raw_path = value.split("=", 1)
        canonical = canonical_variant(variant)
        if canonical not in BUNDLE_PATHS:
            raise ValueError(f"Unknown bundle override variant {variant!r}")
        overrides[canonical] = Path(raw_path).expanduser()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint Freebase graph-variant augmentation for DHN"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/freebase_nc.yaml")
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
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
        help="Override a bundle path, e.g. --bundle exact_2=/path/bundle.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/freebase_nc/data_augmentation"),
    )
    parser.add_argument(
        "--super-epochs", "--epochs", dest="super_epochs", type=int, default=None
    )
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate both bundles and the one-layer DHN contract, then exit",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    variants = parse_variants(args.variants)
    seeds = (
        [int(value) for value in parse_csv(args.seeds)]
        if args.seeds
        else [1566911444, 20241017, 20251017]
    )
    args.workspace_root = args.workspace_root.resolve()
    args.bundle_overrides = parse_bundle_overrides(args.bundle)
    for variant, path in list(args.bundle_overrides.items()):
        if not path.is_absolute():
            args.bundle_overrides[variant] = args.workspace_root / path

    if args.preflight_only:
        bundles = prepare_bundles(
            variants, args.workspace_root, args.bundle_overrides
        )
        validate_model_bundle_contract(config, bundles)
        reference = bundles[variants[0]]
        print(
            f"[OK] Freebase joint augmentation preflight: variants={variants}, "
            f"nodes={reference['data'].num_nodes}, "
            f"labeled={reference['meta']['num_labeled_targets']}, seeds={seeds}"
        )
        for variant in variants:
            mappings = bundles[variant]["data"].mapping_index_dict
            counts = {
                name: 0 if value is None else int(value.shape[0])
                for name, value in mappings.items()
            }
            print(
                f"  {variant}: mappings={counts} "
                f"path={bundles[variant]['_path']}"
            )
        return

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_seed(args, config, variants, seed, output_root) for seed in seeds
    ]
    pd.DataFrame(
        [flatten_seed_summary(summary) for summary in summaries]
    ).to_csv(output_root / "seed_summary.csv", index=False)
    write_json(output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
