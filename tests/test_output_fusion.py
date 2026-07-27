"""Contracts for post-hoc averaging of independent NC baseline logits."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.node_classification.output_fusion import (
    fuse_seed,
    full_logit_kendall_tau,
    write_output_fusion_results,
)


def make_run(logits, parameter_bytes, peak_inference):
    return {
        "y_logits": np.asarray(logits, dtype=np.float32),
        "y_true": np.asarray([0, 1], dtype=np.int64),
        "test_node_ids": np.asarray([10, 20], dtype=np.int64),
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
        "inference_gpu_peak_allocated_bytes": peak_inference,
        "inference_gpu_peak_reserved_bytes": peak_inference + 10,
    }


def test_fusion_averages_aligned_logits_and_uses_hybrid_memory_contract():
    runs = {
        "v1": make_run([[3.0, 1.0], [0.0, 2.0]], 100, 50),
        "v2": make_run([[2.0, 0.0], [1.0, 3.0]], 200, 80),
        # A trained but non-fused variant must not affect fusion memory totals.
        "unused": make_run([[1.0, 0.0], [0.0, 1.0]], 400, 900),
    }
    row, comparisons, arrays = fuse_seed(7, ["v1", "v2"], runs)
    assert row["Accuracy"] == 1.0
    assert row["fusion_parameter_bytes_sum"] == 300.0
    assert row["sequential_inference_gpu_peak_allocated_bytes_max"] == 80.0
    assert row["constituent_training_gpu_peak_allocated_bytes_max"] == 30.0
    assert len(comparisons) == 2
    assert arrays["predictions"].tolist() == [0, 1]
    assert full_logit_kendall_tau(
        arrays["averaged_logits"], arrays["variant_0_logits"]
    ) == 1.0


def test_fusion_writes_auditable_seed_and_summary_artifacts():
    runs = {
        7: {
            "v1": make_run([[3.0, 1.0], [0.0, 2.0]], 100, 50),
            "v2": make_run([[2.0, 0.0], [1.0, 3.0]], 200, 80),
        }
    }
    with TemporaryDirectory() as directory:
        output = Path(directory)
        write_output_fusion_results(output, ["v1", "v2"], runs)
        assert (output / "output_fusion_seed7.npz").is_file()
        assert (output / "output_fusion_raw.csv").is_file()
        assert (output / "seed_summary.csv").is_file()
        assert (output / "output_fusion_summary.csv").is_file()
        assert (output / "fusion_vs_variant.csv").is_file()
        assert (output / "output_fusion_manifest.json").is_file()
