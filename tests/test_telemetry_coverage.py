"""Guard the resource-telemetry contract across every DHN training path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DIRECT_TRAINING_RUNNERS = (
    "experiments/node_classification/train.py",
    "experiments/link_prediction/imdb.py",
    "experiments/link_prediction/dblp.py",
    "experiments/link_prediction/wordnet.py",
    "experiments/node_classification/imdb_augmentation.py",
    "experiments/link_prediction/imdb_augmentation.py",
    "experiments/link_prediction/wordnet_augmentation.py",
    "experiments/original_graph_classification/train.py",
)

SUMMARY_AND_FUSION_RUNNERS = (
    "experiments/node_classification/benchmark_imdb.py",
    "experiments/node_classification/benchmark_freebase.py",
    "experiments/node_classification/output_fusion.py",
    "experiments/link_prediction/wordnet_output_fusion.py",
)


def test_direct_training_runners_measure_both_cuda_phases():
    for relative_path in DIRECT_TRAINING_RUNNERS:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "reset_cuda_peak" in source, relative_path
        assert "cuda_memory_stats" in source, relative_path
        assert "training_gpu" in source, relative_path
        assert "inference_gpu" in source, relative_path
        assert "process_peak_rss_bytes" in source, relative_path


def test_wrappers_and_fusion_outputs_require_resource_schema():
    for relative_path in SUMMARY_AND_FUSION_RUNNERS:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "RESOURCE_METRIC_KEYS" in source, relative_path
