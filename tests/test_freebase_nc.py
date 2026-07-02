"""Focused checks for the Freebase node-classification pipeline."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.node_classification.train import FeaturelessDHNNodeClassifier
from preprocess.freebase.node_classification import (
    masks_from_ids,
    rooted_edge_mappings,
    stratified_split,
)


def test_stratified_split_is_deterministic_and_disjoint():
    node_ids = np.arange(80, dtype=np.int64)
    labels = np.repeat(np.arange(8, dtype=np.int64), 10)
    first = stratified_split(node_ids, labels, 1566911444)
    second = stratified_split(node_ids, labels, 1566911444)

    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    train_ids, val_ids, test_ids = first
    assert (len(train_ids), len(val_ids), len(test_ids)) == (48, 16, 16)
    assert not set(train_ids) & set(val_ids)
    assert not set(train_ids) & set(test_ids)
    assert not set(val_ids) & set(test_ids)


def test_rooted_edges_are_undirected_and_deduplicated():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "link.dat"
        path.write_text(
            "0\t1\t0\t1.0\n"
            "1\t0\t0\t1.0\n"
            "1\t2\t0\t1.0\n"
            "2\t3\t0\t1.0\n"
            "3\t3\t0\t1.0\n"
        )
        mappings = rooted_edge_mappings(
            path, np.asarray([0, 2], dtype=np.int64), max_mappings=10
        )

    assert mappings.tolist() == [[0, 1], [2, 1], [2, 3]]


def test_rooted_edge_limit_refuses_truncation():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "link.dat"
        path.write_text("0\t1\t0\t1.0\n0\t2\t0\t1.0\n")
        try:
            rooted_edge_mappings(
                path, np.asarray([0], dtype=np.int64), max_mappings=1
            )
        except RuntimeError as error:
            assert "Refusing to truncate or sample" in str(error)
        else:
            raise AssertionError("Expected rooted mapping limit to abort")


def test_featureless_dhn_backpropagates_to_embeddings():
    p1 = torch.tensor([[0], [1], [2]], dtype=torch.long)
    c2 = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
    graph = Data(
        x=None,
        y=torch.tensor([0, 1, 0]),
        mapping_index_dict={"p1": p1, "c2": c2},
        num_nodes=3,
    )
    graph.batch = torch.zeros(3, dtype=torch.long)
    graph.batch_size = 1

    model = FeaturelessDHNNodeClassifier(
        num_nodes=3,
        embedding_dim=8,
        out_dim=2,
        layers_config=[{"p1": (8, 4, 1), "c2": (8, 4, 2)}],
        act_module=nn.ReLU,
        agg=None,
        inplace=False,
    )
    output = model(graph)
    nn.CrossEntropyLoss()(output, graph.y).backward()

    assert tuple(output.shape) == (3, 2)
    assert model.node_embedding.weight.grad is not None
    assert torch.isfinite(model.node_embedding.weight.grad).all()


if __name__ == "__main__":
    test_stratified_split_is_deterministic_and_disjoint()
    test_rooted_edges_are_undirected_and_deduplicated()
    test_rooted_edge_limit_refuses_truncation()
    test_featureless_dhn_backpropagates_to_embeddings()
    print("Freebase NC tests passed")
