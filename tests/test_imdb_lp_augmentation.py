"""Contracts for joint IMDb DHN link-prediction augmentation."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.link_prediction.imdb_augmentation import (
    VALID_VARIANTS,
    build_score_frame,
    flatten_seed_summary,
    link_invariance_rows,
    parse_variants,
    prepare_bundles,
    rgcn_metric_names,
    validate_model_bundle_contract,
)


def make_bundle(
    path: Path,
    task: str,
    variant: str,
    split_offset: int = 0,
) -> None:
    # movie=[0,2), director=[2,4), actor=[4,5), link=[5,7), genre=[7,10)
    offsets = {"movie": 0, "director": 2, "actor": 4, "link": 5, "genre": 7}
    counts = {"movie": 2, "director": 2, "actor": 1, "link": 2, "genre": 3}
    p1 = torch.arange(10).reshape(-1, 1)
    c2 = torch.tensor([[0, 4], [4, 0], [1, 4], [4, 1]])
    data = Data(
        x=None,
        edge_index=c2.T,
        mapping_index_dict={"p1": p1, "c2": c2},
        num_nodes=10,
    )
    data.batch = torch.zeros(10, dtype=torch.long)
    data.batch_size = 1
    target_offset = offsets[{"md": "director", "mg": "genre", "ml": "link"}[task]]
    pos = torch.tensor([[0, target_offset], [1, target_offset + 1]])
    neg = torch.tensor(
        [
            [target_offset + 1],
            [target_offset],
        ]
    )
    splits = {
        f"{split}_{kind}": value + split_offset
        for split in ("train", "val", "test")
        for kind, value in (("pos", pos), ("neg", neg))
    }
    torch.save(
        {
            "data": data,
            "splits": splits,
            "node_offsets": offsets,
            "meta": {
                "task": task,
                "variant": variant,
                "num_nodes_per_type": counts,
                "num_nodes_total": 10,
                "neg_k": 1,
                "patterns": ["p1", "c2"],
            },
        },
        path,
    )


def test_task_specific_variant_contract():
    assert VALID_VARIANTS["md"] == ("v1", "v3")
    assert parse_variants("md", "v1,v3") == ["v1", "v3"]
    try:
        parse_variants("md", "v1,v2")
    except ValueError as error:
        assert "Invalid variants for md" in str(error)
    else:
        raise AssertionError("md/v2 must be rejected")


def test_preflight_requires_identical_fixed_candidate_tables():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        overrides = {}
        for variant in ("v1", "v2"):
            path = root / f"{variant}.pt"
            make_bundle(path, "mg", variant)
            overrides[variant] = path
        bundles = prepare_bundles("mg", ["v1", "v2"], root, overrides)
        config = {
            "model": {
                "layers_config": [
                    {"p1": [-1, 4, 1], "c2": [-1, 4, 2]},
                    {"p1": [-1, 4, 1], "c2": [-1, 4, 2]},
                ]
            }
        }
        validate_model_bundle_contract(config, bundles)

        make_bundle(overrides["v2"], "mg", "v2", split_offset=1)
        try:
            prepare_bundles("mg", ["v1", "v2"], root, overrides)
        except ValueError as error:
            assert "train_pos differs" in str(error)
        else:
            raise AssertionError("Candidate-table mismatch should fail preflight")


def test_score_frame_rank_invariance_and_flat_summary_schema():
    test_pos = torch.tensor([[0, 2], [1, 3]])
    test_neg = torch.tensor([[3, 3], [2, 2]])
    positive = torch.tensor([2.0, 0.0])
    negative = torch.tensor([[1.0, 3.0], [-1.0, -2.0]])
    offsets = {"movie": 0, "director": 2}
    frame = build_score_frame(
        "md", test_pos, test_neg, positive, negative, offsets
    )
    assert list(frame[frame["label"] == 1]["rank_of_positive"]) == [2, 1]
    assert "director_local" in frame
    assert list(frame["candidate_id"]) == list(range(6))
    rows = link_invariance_rows({"v1": frame, "v3": frame.copy()}, 0.5)
    assert rows[0]["prediction_agreement"] == 1.0
    assert rows[0]["rank_agreement"] == 1.0
    assert rgcn_metric_names({"auc": 0.5, "hits@3": 0.75}, 2) == {
        "AUC": 0.5,
        "Hits@3": 0.75,
        "ranking_queries": 2.0,
    }

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
            "task": "md",
            "seed": 1,
            "variants": ["v1", "v3"],
            "training_seconds": 2.0,
            "best_mean_val_loss": 0.4,
            "epoch_accounting": {
                "batch_size": 0,
                "effective_batch_size": 10,
                "batches_per_variant": 1,
                "super_epochs_ran": 3,
                "variant_epochs_ran": 6,
                "optimizer_steps": 6,
            },
            "mean_test_metrics": {"AUC": 0.5, "MRR": 0.4},
            "memory": memory,
        }
    )
    assert row["task"] == "md"
    assert row["mean_test_AUC"] == 0.5
    assert row["training_gpu_peak_allocated_bytes"] == 3
    assert row["process_peak_rss_bytes"] == 30
