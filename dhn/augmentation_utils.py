"""Runtime and artifact helpers for shared graph-variant experiments.

The output keys deliberately match INV-GNN's augmentation runners so DHN and
RGCN results can be aggregated with the same reporting code.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


CUDA_MEMORY_KEYS = (
    "gpu_allocated_bytes",
    "gpu_reserved_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
)
RESOURCE_METRIC_KEYS = (
    "parameter_bytes",
    "buffer_bytes",
    "static_model_bytes",
    "checkpoint_bytes",
    "process_peak_rss_bytes",
    *(f"training_{key}" for key in CUDA_MEMORY_KEYS),
    *(f"inference_{key}" for key in CUDA_MEMORY_KEYS),
)


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[warn] {device} requested but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return device


class EarlyStopper:
    def __init__(self, mode: str, patience: int, min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf if mode == "min" else -math.inf
        self.bad_cycles = 0

    def update(self, value: float) -> bool:
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = float(value)
            self.bad_cycles = 0
        else:
            self.bad_cycles += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_cycles >= self.patience


def model_memory_bytes(model: nn.Module) -> dict[str, int]:
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "static_model_bytes": int(parameter_bytes + buffer_bytes),
    }


def process_peak_rss_bytes() -> int:
    """Peak resident set size for this process, including loaded graph bundles."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def reset_cuda_peak(device: torch.device) -> None:
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {key: 0 for key in CUDA_MEMORY_KEYS}
    torch.cuda.synchronize(device)
    return {
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def merge_cuda_memory_stats(
    previous: Optional[Mapping[str, int]], current: Mapping[str, int]
) -> dict[str, int]:
    previous = previous or {}
    return {
        "gpu_allocated_bytes": int(current.get("gpu_allocated_bytes", 0)),
        "gpu_reserved_bytes": int(current.get("gpu_reserved_bytes", 0)),
        "gpu_peak_allocated_bytes": max(
            int(previous.get("gpu_peak_allocated_bytes", 0)),
            int(current.get("gpu_peak_allocated_bytes", 0)),
        ),
        "gpu_peak_reserved_bytes": max(
            int(previous.get("gpu_peak_reserved_bytes", 0)),
            int(current.get("gpu_peak_reserved_bytes", 0)),
        ),
    }


def checkpoint_size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def flat_resource_metrics(
    model: nn.Module,
    *,
    training_gpu: Mapping[str, int],
    inference_gpu: Mapping[str, int],
    checkpoint_path: Optional[Path] = None,
    peak_rss_bytes: Optional[int] = None,
) -> dict[str, int]:
    """Flatten the augmentation resource contract for legacy CSV writers."""
    metrics = {
        **model_memory_bytes(model),
        "checkpoint_bytes": (
            checkpoint_size_bytes(Path(checkpoint_path))
            if checkpoint_path is not None
            else 0
        ),
        "process_peak_rss_bytes": (
            process_peak_rss_bytes()
            if peak_rss_bytes is None
            else int(peak_rss_bytes)
        ),
    }
    for phase, values in (
        ("training", training_gpu),
        ("inference", inference_gpu),
    ):
        for key in CUDA_MEMORY_KEYS:
            metrics[f"{phase}_{key}"] = int(values.get(key, 0))
    return metrics


def torch_load_full(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mean_dict(metric_dicts: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts]))
        for key in keys
    }


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor
) -> dict[str, float]:
    selected = logits[indices]
    y_true = labels[indices].detach().cpu().numpy()
    y_pred = selected.argmax(dim=1).detach().cpu().numpy()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Micro_F1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "Macro_F1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def classification_invariance_rows(
    outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    variants = list(outputs)
    rows: list[dict[str, Any]] = []
    for index, variant_a in enumerate(variants):
        for variant_b in variants[index + 1 :]:
            left = outputs[variant_a]
            right = outputs[variant_b]
            if not np.array_equal(left["item_id"], right["item_id"]):
                raise ValueError(
                    f"Test item IDs differ between {variant_a} and {variant_b}"
                )
            if left["logits"].shape != right["logits"].shape:
                raise ValueError(
                    f"Logit shapes differ between {variant_a} and {variant_b}"
                )
            difference = left["logits"] - right["logits"]
            rows.append(
                {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "kendall_tau_flat_logits": _safe_tau(
                        left["logits"].ravel(), right["logits"].ravel()
                    ),
                    "kendall_tau_confidence": _safe_tau(
                        left["confidence"], right["confidence"]
                    ),
                    "prediction_agreement": float(
                        np.mean(left["prediction"] == right["prediction"])
                    ),
                    "max_abs_logit_diff": float(
                        np.max(np.abs(difference))
                    ),
                    "mean_abs_logit_diff": float(
                        np.mean(np.abs(difference))
                    ),
                    "mean_l2_logit_diff": float(
                        np.linalg.norm(difference, axis=1).mean()
                    ),
                }
            )
    return rows


def link_binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "AP": float(average_precision_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _safe_tau(a: np.ndarray, b: np.ndarray) -> float:
    tau = kendalltau(np.asarray(a), np.asarray(b), nan_policy="omit").statistic
    return float(tau) if tau is not None and np.isfinite(tau) else float("nan")


def triple_link_invariance_rows(
    outputs: Mapping[str, pd.DataFrame], threshold: float
) -> list[dict[str, Any]]:
    variants = list(outputs)
    rows: list[dict[str, Any]] = []
    keys = ["query_id", "head", "relation", "tail", "label"]
    for index, variant_a in enumerate(variants):
        for variant_b in variants[index + 1 :]:
            left = outputs[variant_a]
            right = outputs[variant_b]
            merged = left.merge(
                right, on=keys, suffixes=("_a", "_b"), how="inner"
            )
            if len(merged) != len(left) or len(merged) != len(right):
                raise ValueError(
                    f"Candidate sets do not align for {variant_a} and {variant_b}"
                )
            score_a = merged["score_a"].to_numpy()
            score_b = merged["score_b"].to_numpy()
            difference = score_a - score_b
            rows.append(
                {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "kendall_tau_scores": _safe_tau(score_a, score_b),
                    "prediction_agreement": float(
                        np.mean(
                            (score_a >= threshold) == (score_b >= threshold)
                        )
                    ),
                    "max_abs_score_diff": float(np.max(np.abs(difference))),
                    "mean_abs_score_diff": float(np.mean(np.abs(difference))),
                }
            )
    return rows


def _capture_local_numpy_rng(rng) -> dict[str, Any]:
    if isinstance(rng, np.random.Generator):
        return {"kind": "generator", "state": rng.bit_generator.state}
    if isinstance(rng, np.random.RandomState):
        return {"kind": "random_state", "state": rng.get_state()}
    raise TypeError(f"Unsupported NumPy RNG type: {type(rng)!r}")


def capture_rng_state(rng) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_local": _capture_local_numpy_rng(rng),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_local_numpy_rng(saved: Mapping[str, Any], rng) -> None:
    kind = saved.get("kind")
    if kind == "generator" and isinstance(rng, np.random.Generator):
        rng.bit_generator.state = saved["state"]
    elif kind == "random_state" and isinstance(rng, np.random.RandomState):
        rng.set_state(saved["state"])
    else:
        raise ValueError(
            f"Saved NumPy RNG kind {kind!r} is incompatible with {type(rng)!r}"
        )


def restore_rng_state(state: Mapping[str, Any], rng) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    _restore_local_numpy_rng(state["numpy_local"], rng)
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def _early_stopper_state(stopper: EarlyStopper) -> dict[str, Any]:
    return {
        "mode": stopper.mode,
        "patience": stopper.patience,
        "min_delta": stopper.min_delta,
        "best": stopper.best,
        "bad_cycles": stopper.bad_cycles,
    }


def _restore_early_stopper(
    stopper: EarlyStopper, state: Mapping[str, Any]
) -> None:
    expected = (stopper.mode, stopper.patience, stopper.min_delta)
    actual = (
        state["mode"],
        int(state["patience"]),
        float(state["min_delta"]),
    )
    if expected != actual:
        raise ValueError(f"Early-stopper configuration differs: {expected} vs {actual}")
    stopper.best = float(state["best"])
    stopper.bad_cycles = int(state["bad_cycles"])


def _canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(json_ready(dict(config)), sort_keys=True, separators=(",", ":"))


def optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_latest_training_state(
    path: Path,
    *,
    resume: bool,
    run_config: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopper: EarlyStopper,
    rng,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        existing = [
            candidate
            for candidate in (
                path.parent / "shared_checkpoint.pt",
                path.parent / "training_history.csv",
            )
            if candidate.exists()
        ]
        if existing:
            raise RuntimeError(
                "This seed directory already has non-resumable outputs but no "
                f"{path.name}. Move it or choose a new --output-dir."
            )
        if resume:
            print(f"[resume] No state at {path}; starting a new run.")
        return None
    if not resume:
        raise RuntimeError(
            f"Existing state found at {path}; pass --resume or use a new output."
        )
    state = torch_load_full(path, map_location=device)
    if int(state.get("state_version", 0)) != 1:
        raise ValueError(f"Unsupported resume-state version in {path}")
    if _canonical_config(state["run_config"]) != _canonical_config(run_config):
        raise ValueError(
            "Resume configuration differs. Only --super-epochs and --device "
            "may change when resuming."
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    optimizer_to_device(optimizer, device)
    _restore_early_stopper(early_stopper, state["early_stopper"])
    restore_rng_state(state["rng_state"], rng)
    print(
        f"[resume] Restored completed super-epoch "
        f"{state['completed_super_epoch']} from {path}."
    )
    return state


def save_latest_training_state(
    path: Path,
    *,
    dataset: str = "WORDNET",
    run_config: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopper: EarlyStopper,
    rng,
    completed_super_epoch: int,
    counters: Mapping[str, int],
    history: Sequence[Mapping[str, Any]],
    training_seconds_elapsed: float,
    process_peak_rss_bytes_value: int,
    training_gpu: Mapping[str, int],
) -> None:
    atomic_torch_save(
        {
            "state_version": 1,
            "dataset": dataset,
            "run_config": dict(run_config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "early_stopper": _early_stopper_state(early_stopper),
            "rng_state": capture_rng_state(rng),
            "completed_super_epoch": int(completed_super_epoch),
            "counters": {str(k): int(v) for k, v in counters.items()},
            "history": [dict(row) for row in history],
            "training_seconds_elapsed": float(training_seconds_elapsed),
            "process_peak_rss_bytes": int(process_peak_rss_bytes_value),
            "training_gpu": {str(k): int(v) for k, v in training_gpu.items()},
        },
        path,
    )
