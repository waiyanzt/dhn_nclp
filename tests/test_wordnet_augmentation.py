"""Focused contracts for the four-variant WordNet DHN workflow."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhn.augmentation_utils import cuda_memory_stats
from experiments.link_prediction.wordnet_augmentation import (
    DEFAULT_VARIANTS,
    VARIANTS,
    canonicalize_variants,
    fixed_tail_candidates,
    flatten_seed_summary,
)
from preprocess.wordnet.augmentation import FORMAT_VERSION, make_mapping_tensors


def test_four_canonical_variants_and_aliases():
    assert VARIANTS == (
        "no_changes",
        "all_inverse_edges",
        "transitive_edges",
        "universal_edges",
    )
    assert canonicalize_variants("unchanged,inverse,transitive,universal") == list(
        VARIANTS
    )
    assert DEFAULT_VARIANTS == (
        "no_changes",
        "all_inverse_edges",
        "universal_edges",
    )
    assert "transitive_edges" not in DEFAULT_VARIANTS
    assert FORMAT_VERSION == "wordnet_lp_four_variants_v1"


def test_mapping_tensors_match_legacy_p1_c2_semantics():
    triples = np.asarray(
        [
            [0, 0, 1],
            [1, 1, 0],  # relation/parallel direction collapses
            [1, 0, 2],
            [2, 0, 2],  # self-loop does not form a c2 mapping
        ],
        dtype=np.int64,
    )
    edge_index, mappings, undirected_count = make_mapping_tensors(triples, 4)
    assert mappings["p1"].tolist() == [[0], [1], [2], [3]]
    assert mappings["c2"].tolist() == [[1, 0], [2, 1], [0, 1], [1, 2]]
    assert edge_index.tolist() == [[1, 2, 0, 1], [0, 1, 1, 2]]
    assert undirected_count == 2


def test_fixed_candidates_are_deterministic_and_exclude_known_triples():
    positives = np.asarray([[0, 0, 1], [1, 0, 2]], dtype=np.int64)
    known = {1, 1 * 4 + 2}  # N=4, R=1 packed keys
    first = fixed_tail_candidates(positives, known, 4, 1, 2, 123)
    second = fixed_tail_candidates(positives, known, 4, 1, 2, 123)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    triples, labels, query_ids = first
    assert triples.shape == (6, 3)
    assert labels.tolist() == [1, 0, 0, 1, 0, 0]
    assert query_ids.tolist() == [0, 0, 0, 1, 1, 1]
    assert all(tuple(row) not in {(0, 0, 1), (1, 0, 2)} for row in triples[labels == 0])


def test_cpu_memory_and_flattened_seed_summary_contract():
    import torch

    assert set(cuda_memory_stats(torch.device("cpu"))) == {
        "gpu_allocated_bytes",
        "gpu_reserved_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
    }
    summary = {
        "seed": 1,
        "variants": list(DEFAULT_VARIANTS),
        "training_seconds": 2.0,
        "best_mean_val_filtered_MRR": 0.3,
        "epoch_accounting": {
            "super_epochs_ran": 2,
            "variant_epochs_ran": 8,
            "updates_per_super_epoch": 4,
            "optimizer_steps": 8,
        },
        "mean_legacy_test_metrics": {"filtered_MRR": 0.2},
        "mean_shared_candidate_metrics": {"AUC": 0.6},
        "memory": {
            "parameter_bytes": 10,
            "buffer_bytes": 0,
            "static_model_bytes": 10,
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
    assert row["variants"] == "no_changes,all_inverse_edges,universal_edges"
    assert row["mean_legacy_test_filtered_MRR"] == 0.2
    assert row["process_peak_rss_bytes"] == 30
    assert row["training_gpu_peak_allocated_bytes"] == 3
    assert row["inference_gpu_peak_reserved_bytes"] == 8
