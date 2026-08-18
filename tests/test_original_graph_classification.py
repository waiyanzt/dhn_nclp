"""Contracts for the restored original DHN graph-classification path."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch_geometric.data import Data

from dhn.augmentation_utils import RESOURCE_METRIC_KEYS
from dhn.datasets import HomDataLoader, HomDataset


def test_cached_hom_dataset_and_loader_contract():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        samples = [
            Data(
                x=torch.ones(2, 3),
                y=torch.tensor([1]),
                mapping_index_dict={
                    "c2": torch.tensor([[0, 1], [1, 0]])
                },
            ),
            Data(
                x=torch.ones(1, 3),
                y=torch.tensor([0]),
                mapping_index_dict={"c2": None},
            ),
        ]
        torch.save((2, 3, samples), root / "fixture.pt")

        dataset = HomDataset(name="fixture", root_path=str(root))
        assert len(dataset) == 2
        assert dataset.num_classes == 2
        assert dataset.num_features == 3

        batch = next(iter(HomDataLoader(dataset, batch_size=2)))
        assert tuple(batch.x.shape) == (3, 3)
        assert batch.y.tolist() == [1, 0]
        assert batch.batch_size == 2
        assert batch.mapping_index_dict["c2"].tolist() == [[0, 1], [1, 0]]


def test_graph_classification_telemetry_contract_is_complete():
    expected = {
        "parameter_bytes",
        "buffer_bytes",
        "static_model_bytes",
        "checkpoint_bytes",
        "process_peak_rss_bytes",
        "training_gpu_allocated_bytes",
        "training_gpu_reserved_bytes",
        "training_gpu_peak_allocated_bytes",
        "training_gpu_peak_reserved_bytes",
        "inference_gpu_allocated_bytes",
        "inference_gpu_reserved_bytes",
        "inference_gpu_peak_allocated_bytes",
        "inference_gpu_peak_reserved_bytes",
    }
    assert set(RESOURCE_METRIC_KEYS) == expected
