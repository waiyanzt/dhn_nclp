# Deep Homomorphism Networks — Node Classification

This repository contains a refactored implementation of **Deep Homomorphism Networks (DHN)**, originally introduced by Hoang NT and Takanori Maehara in their NeurIPS 2024 paper:

> Hoang NT and Takanori Maehara. "Deep Homomorphism Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

The original work targets graph classification. This implementation adapts the DHN architecture for **node classification**, with the goal of benchmarking the method on an in-house graph dataset.

---

## Method Overview

DHN computes graph representations by counting homomorphisms from a set of pattern graphs (cycles and cliques) into the input graph. For each node, the contribution of each homomorphism is computed via learned, pattern-specific transformations and aggregated back onto the node. This gives each node a representation that is aware of the local and global subgraph structure around it, going beyond standard message-passing GNNs in expressive power.

The key components are:

- **`HomConv` (`layers.py`)** — the core convolution operator. For a given pattern (e.g., a cycle of length 4 or a 3-clique), it applies a learned transformation to every homomorphism of that pattern into the graph and scatters the result back to the root node.

- **`DHN` (`models.py`)** — the full model. Stacks multiple `HomConv` layers, each operating over a configurable set of cycle and clique patterns. The original aggregation module for graph-level readout is retained but will be removed or bypassed for node-level prediction.

- **`graph_enumerations.py`** — precomputes cycle and clique homomorphism mappings for a graph using NetworkX. Results are stored as index tensors for efficient batched lookup during forward passes.

- **`HomDataset` / `HomDataLoader` (`datasets.py`)** — handles data loading and preprocessing. Homomorphism mappings are computed once during dataset construction and cached to disk. A custom collate function (`hom_collate`) handles variable-length mapping tensors across batched graphs.

- **`utils.py`** — builder utilities for layers, activations, optimizers, and learning rate schedulers.

---

---

## Citation

If you use this code or build on the DHN method, please cite the original authors:

```bibtex
@inproceedings{nt2024deephomomorphism,
  title     = {Deep Homomorphism Networks},
  author    = {Hoang NT and Takanori Maehara},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
