"""Contracts for independent WordNet LP raw-score output fusion."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.link_prediction.wordnet_output_fusion import (
    candidate_metrics,
    fuse_seed,
)


def make_run(parameter_bytes, inference_peak):
    return {
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": 4,
        "static_model_bytes": parameter_bytes + 4,
        "checkpoint_bytes": parameter_bytes + 20,
        "process_peak_rss_bytes": 1000 + parameter_bytes,
        "training_gpu_allocated_bytes": 10,
        "training_gpu_reserved_bytes": 20,
        "training_gpu_peak_allocated_bytes": 30,
        "training_gpu_peak_reserved_bytes": 40,
        "inference_gpu_allocated_bytes": 5,
        "inference_gpu_reserved_bytes": 10,
        "inference_gpu_peak_allocated_bytes": inference_peak,
        "inference_gpu_peak_reserved_bytes": inference_peak + 10,
    }


def make_artifact(scores):
    return {
        "test_pos": np.asarray([[1, 0, 3], [2, 0, 4]], dtype=np.int64),
        "candidate_ids": np.asarray([[0, 2, 3], [0, 1, 4]], dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.uint8),
        "candidate_rng_seed": np.asarray(42),
    }


def test_candidate_metrics_rank_the_single_true_tail_per_query():
    metrics = candidate_metrics(
        np.asarray([[0.0, 1.0, 3.0], [0.0, 2.0, 1.0]]),
        np.asarray([[0, 0, 1], [0, 0, 1]]),
        hits_k=(1, 3),
    )
    assert metrics["candidate_MRR_tail"] == 0.75
    assert metrics["candidate_Hits@1_tail"] == 0.5
    assert metrics["candidate_Hits@3_tail"] == 1.0


def test_wordnet_fusion_averages_raw_scores_and_hybrid_memory():
    artifacts = {
        "no_changes": make_artifact(
            [[0.0, 1.0, 4.0], [0.0, 1.0, 3.0]]
        ),
        "all_inverse_edges": make_artifact(
            [[1.0, 0.0, 3.0], [1.0, 0.0, 4.0]]
        ),
    }
    runs = {
        "no_changes": make_run(100, 50),
        "all_inverse_edges": make_run(200, 80),
    }
    row, comparisons, arrays = fuse_seed(
        7,
        ["no_changes", "all_inverse_edges"],
        artifacts,
        runs,
        hits_k=(1, 3),
    )
    assert row["candidate_MRR_tail"] == 1.0
    assert row["fusion_parameter_bytes_sum"] == 300.0
    assert row["sequential_inference_gpu_peak_allocated_bytes_max"] == 80.0
    assert len(comparisons) == 2
    assert arrays["averaged_scores"].shape == (2, 3)
