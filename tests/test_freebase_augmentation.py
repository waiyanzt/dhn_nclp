"""Contracts for joint Freebase DHN graph-variant augmentation."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch_geometric.data import Data

from experiments.node_classification.freebase_augmentation import (
    create_model,
    flatten_seed_summary,
    parse_variants,
    prepare_bundles,
    validate_model_bundle_contract,
)


def make_bundle(path: Path, variant: str, c2: torch.Tensor) -> None:
    num_nodes = 6
    labels = torch.tensor([0, 1, 0, 1, -1, -1], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False, False, False])
    val_mask = torch.tensor([False, False, True, False, False, False])
    test_mask = torch.tensor([False, False, False, True, False, False])
    p1 = torch.arange(4, dtype=torch.long).reshape(-1, 1)
    data = Data(
        x=None,
        y=labels,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        mapping_index_dict={"p1": p1, "c2": c2},
        num_nodes=num_nodes,
    )
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    data.batch_size = 1
    torch.save(
        {
            "data": data,
            "num_features": 8,
            "num_classes": 2,
            "meta": {
                "dataset": "Freebase",
                "variant": variant,
                "patterns": ["p1", "c2"],
                "mapping_scope": "labeled_target_roots",
                "num_labeled_targets": 4,
            },
        },
        path,
    )


def make_config() -> dict:
    return {
        "model": {
            "learned_node_embeddings": True,
            "layers_config": [
                {"p1": [-1, 4, 1], "c2": [-1, 4, 2]}
            ],
            "agg": None,
            "activation": {"name": "ReLU", "kwargs": {"inplace": False}},
            "homconv_kwargs": {
                "mapping_chunk_size": 2,
                "checkpoint_chunks": True,
            },
        },
        "training": {"lr_scheduling": {"name": None}},
    }


def test_aliases_and_shared_bundle_contract():
    assert parse_variants("no_changes,exact2") == ["unchanged", "exact_2"]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        unchanged = root / "unchanged.pt"
        exact_2 = root / "exact_2.pt"
        make_bundle(
            unchanged,
            "unchanged",
            torch.tensor([[0, 4], [1, 5]], dtype=torch.long),
        )
        make_bundle(
            exact_2,
            "exact_2",
            torch.tensor([[0, 5], [1, 4], [2, 5]], dtype=torch.long),
        )
        bundles = prepare_bundles(
            ["unchanged", "exact_2"],
            root,
            {"unchanged": unchanged, "exact_2": exact_2},
        )
        validate_model_bundle_contract(make_config(), bundles)
        assert bundles["unchanged"]["data"].x is None
        assert not torch.equal(
            bundles["unchanged"]["data"].mapping_index_dict["c2"],
            bundles["exact_2"]["data"].mapping_index_dict["c2"],
        )


def test_featureless_shared_model_backpropagates():
    config = make_config()
    model = create_model(config, 6, 8, 2)
    graph = Data(
        x=None,
        y=torch.tensor([0, 1, 0, 1, -1, -1]),
        mapping_index_dict={
            "p1": torch.arange(4).reshape(-1, 1),
            "c2": torch.tensor([[0, 4], [1, 5], [2, 4], [3, 5]]),
        },
        num_nodes=6,
    )
    graph.batch = torch.zeros(6, dtype=torch.long)
    graph.batch_size = 1
    logits = model(graph)
    nn.CrossEntropyLoss()(logits[:2], graph.y[:2]).backward()
    assert tuple(logits.shape) == (6, 2)
    assert model.node_embedding.weight.grad is not None


def test_multilayer_config_is_rejected_for_rooted_mappings():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        paths = {}
        for variant in ("unchanged", "exact_2"):
            path = root / f"{variant}.pt"
            make_bundle(
                path,
                variant,
                torch.tensor([[0, 4], [1, 5]], dtype=torch.long),
            )
            paths[variant] = path
        bundles = prepare_bundles(
            ["unchanged", "exact_2"], root, paths
        )
        config = make_config()
        config["model"]["layers_config"].append(
            {"p1": [-1, 4, 1], "c2": [-1, 4, 2]}
        )
        try:
            validate_model_bundle_contract(config, bundles)
        except ValueError as error:
            assert "exactly one model layer" in str(error)
        else:
            raise AssertionError("Multilayer target-root-only DHN must fail")


def test_flattened_summary_contains_table_and_memory_fields():
    summary = {
        "seed": 1,
        "variants": ["unchanged", "exact_2"],
        "training_seconds": 2.0,
        "best_mean_val_loss": 0.4,
        "epoch_accounting": {
            "super_epochs_ran": 3,
            "variant_epochs_ran": 6,
            "updates_per_super_epoch": 2,
            "optimizer_steps": 6,
        },
        "mean_test_metrics": {
            "Accuracy": 0.6,
            "Precision_macro": 0.5,
            "Recall_macro": 0.4,
            "Micro_F1": 0.6,
            "Macro_F1": 0.45,
        },
        "memory": {
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
        },
    }
    row = flatten_seed_summary(summary)
    assert row["variants"] == "unchanged,exact_2"
    assert row["mean_test_Accuracy"] == 0.6
    assert row["training_gpu_peak_allocated_bytes"] == 3
    assert row["inference_gpu_peak_reserved_bytes"] == 8
    assert row["process_peak_rss_bytes"] == 30
