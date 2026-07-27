"""Contracts for joint IMDb DHN node-classification augmentation."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhn.augmentation_utils import classification_invariance_rows
from experiments.node_classification.imdb_augmentation import (
    DEFAULT_VARIANTS,
    flatten_seed_summary,
    parse_variants,
    prepare_bundles,
    validate_model_bundle_contract,
)


def make_bundle(path: Path, variant: str, feature_offset: float = 0.0) -> None:
    features = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    features = features + feature_offset
    labels = torch.tensor([0, 1, 0, -1])
    p1 = torch.arange(4).reshape(-1, 1)
    c2 = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    data = Data(
        x=features,
        y=labels,
        edge_index=c2.T,
        mapping_index_dict={"p1": p1, "c2": c2},
        train_mask=torch.tensor([True, False, False, False]),
        val_mask=torch.tensor([False, True, False, False]),
        test_mask=torch.tensor([False, False, True, False]),
        num_nodes=4,
    )
    data.batch = torch.zeros(4, dtype=torch.long)
    data.batch_size = 1
    torch.save(
        {
            "data": data,
            "num_features": 3,
            "num_classes": 2,
            "meta": {"variant": variant, "patterns": ["p1", "c2"]},
        },
        path,
    )


def test_variant_aliases_cover_the_four_joint_graphs():
    assert DEFAULT_VARIANTS == ("v1", "v2", "v3", "v4")
    assert parse_variants("IMDb1,IMDb2,IMDb3,IMDb4") == list(DEFAULT_VARIANTS)


def test_bundle_preflight_requires_shared_features_and_supervision():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        overrides = {}
        for variant in DEFAULT_VARIANTS:
            path = root / f"{variant}.pt"
            make_bundle(path, variant)
            overrides[variant] = path
        bundles = prepare_bundles(list(DEFAULT_VARIANTS), root, overrides)
        config = {
            "model": {
                "layers_config": [
                    {"p1": [-1, 4, 1], "c2": [-1, 4, 2]},
                    {"p1": [-1, 4, 1], "c2": [-1, 4, 2]},
                ]
            }
        }
        validate_model_bundle_contract(config, bundles)

        make_bundle(overrides["v4"], "v4", feature_offset=1.0)
        try:
            prepare_bundles(list(DEFAULT_VARIANTS), root, overrides)
        except ValueError as error:
            assert "features differs" in str(error)
        else:
            raise AssertionError("Feature mismatch should fail joint preflight")


def test_invariance_and_seed_summary_match_reporting_contract():
    outputs = {
        "v1": {
            "item_id": np.asarray([1, 2]),
            "logits": np.asarray([[2.0, 1.0], [0.5, 1.5]]),
            "prediction": np.asarray([0, 1]),
            "confidence": np.asarray([0.73, 0.73]),
        },
        "v2": {
            "item_id": np.asarray([1, 2]),
            "logits": np.asarray([[1.8, 1.2], [0.4, 1.6]]),
            "prediction": np.asarray([0, 1]),
            "confidence": np.asarray([0.65, 0.77]),
        },
    }
    rows = classification_invariance_rows(outputs)
    assert len(rows) == 1
    assert rows[0]["prediction_agreement"] == 1.0

    memory = {
        "parameter_bytes": 10,
        "buffer_bytes": 1,
        "static_model_bytes": 11,
        "checkpoint_bytes": 20,
        "process_peak_rss_bytes": 30,
        "training_gpu": {
            "gpu_allocated_bytes": 1,
            "gpu_reserved_bytes": 2,
            "gpu_peak_allocated_bytes": 3,
            "gpu_peak_reserved_bytes": 4,
        },
        "inference_gpu": {
            "gpu_allocated_bytes": 5,
            "gpu_reserved_bytes": 6,
            "gpu_peak_allocated_bytes": 7,
            "gpu_peak_reserved_bytes": 8,
        },
    }
    row = flatten_seed_summary(
        {
            "seed": 1,
            "training_seconds": 2.0,
            "best_mean_val_macro_f1": 0.4,
            "epoch_accounting": {
                "super_epochs_ran": 3,
                "variant_epochs_ran": 12,
                "optimizer_steps": 12,
            },
            "mean_test_metrics": {"Accuracy": 0.5, "Macro_F1": 0.4},
            "memory": memory,
        }
    )
    assert row["super_epochs_ran"] == 3
    assert row["mean_test_Accuracy"] == 0.5
    assert row["training_gpu_peak_allocated_bytes"] == 3
    assert row["process_peak_rss_bytes"] == 30
