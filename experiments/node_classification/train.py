"""Train DHN for node classification on a precomputed graph bundle."""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from dhn.datasets import NodeClassDataset
from dhn.models import DHN
from dhn.utils import (
    get_act_module,
    get_criterion,
    get_lr_scheduler,
    get_optimizer,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


def synchronize_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args():
    p = argparse.ArgumentParser(description="Train DHN for node classification")
    p.add_argument("--config", type=str, default="configs/imdb_nc.yaml")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def move_graph_to(graph, device):
    """Move PyG Data + nested mapping_index_dict to `device`."""
    graph = graph.to(device)
    graph.mapping_index_dict = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in graph.mapping_index_dict.items()
    }
    return graph


def resolve_layers_config(layers_config, feat_dim):
    """Replace any indim == -1 with the running output dim (feat_dim for layer 0)."""
    out = []
    prev_out = feat_dim
    for layer in layers_config:
        new_layer = {}
        layer_out = 0
        for kernel_name, vals in layer.items():
            indim, outdim, ks = vals
            if indim == -1:
                indim = prev_out
            new_layer[kernel_name] = (indim, outdim, ks)
            layer_out += outdim
        out.append(new_layer)
        prev_out = layer_out
    return out


class FeaturelessDHNNodeClassifier(nn.Module):
    """DHN node classifier backed by a learned embedding for every node."""

    def __init__(self, num_nodes, embedding_dim, **dhn_kwargs):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        self.backbone = DHN(**dhn_kwargs)

    def forward(self, graph):
        previous_x = graph.x
        graph.x = self.node_embedding.weight
        try:
            return self.backbone(graph)
        finally:
            graph.x = previous_x


@torch.no_grad()
def evaluate(model, graph, mask, criterion):
    model.eval()
    out = model(graph)
    if mask.sum() == 0:
        return 0.0, 0.0
    loss = criterion(out[mask], graph.y[mask]).item()
    acc = (out[mask].argmax(1) == graph.y[mask]).float().mean().item()
    return acc, loss


def run_once(config, seed=None, data_path=None, logdir=None, verbose=True):
    """Train once and return metrics/artifacts for benchmarking."""
    total_start = time.perf_counter()
    config = dict(config)

    if seed is None:
        seed = config["seed"]
    set_seed(seed)

    device = config["device"]
    if "cuda" in device and not torch.cuda.is_available():
        print(f"WARNING: requested {device} but CUDA unavailable; falling back to CPU")
        device = "cpu"

    if data_path is None:
        data_path = config["data"]["path"]

    if logdir is None:
        logdir = os.path.join(
            config["logging"]["path"], config["logging"]["experiment"]
        )

    logger = SummaryWriter(log_dir=logdir) if SummaryWriter is not None else None

    if verbose:
        print(f"Logging to {logdir}")
        if logger is None:
            print("TensorBoard is not installed; scalar logging is disabled.")

    ds = NodeClassDataset(data_path)
    graph = move_graph_to(ds.data, device)
    feat_dim = ds.num_features
    num_classes = ds.num_classes

    if verbose:
        print(
            f"Loaded data: {graph.num_nodes} nodes, {feat_dim} features, {num_classes} classes"
        )
        print(
            f"Splits: train={int(graph.train_mask.sum())} "
            f"val={int(graph.val_mask.sum())} test={int(graph.test_mask.sum())}"
        )

    layers_config = resolve_layers_config(config["model"]["layers_config"], feat_dim)
    activation_kwargs = config["model"]["activation"].get("kwargs", {})
    homconv_kwargs = config["model"].get("homconv_kwargs", {})
    dhn_kwargs = dict(
        out_dim=num_classes,
        layers_config=layers_config,
        act_module=get_act_module(config["model"]["activation"]["name"]),
        agg=config["model"]["agg"],
        **activation_kwargs,
        **homconv_kwargs,
    )
    if config["model"].get("learned_node_embeddings", False):
        model = FeaturelessDHNNodeClassifier(
            num_nodes=graph.num_nodes,
            embedding_dim=feat_dim,
            **dhn_kwargs,
        ).to(device)
    else:
        model = DHN(**dhn_kwargs).to(device)

    if verbose:
        print(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {n_params}")

    criterion_fn = get_criterion(config["training"]["loss"]["name"])
    criterion = criterion_fn(**config["training"]["loss"].get("kwargs", {}))

    optimizer_fn = get_optimizer(config["training"]["optimizer"]["name"])
    optimizer = optimizer_fn(
        model.parameters(),
        **config["training"]["optimizer"].get("kwargs", {}),
    )

    scheduler = None
    sched_cfg = config["training"].get("lr_scheduling", {})
    if sched_cfg.get("name"):
        sched_fn = get_lr_scheduler(sched_cfg["name"])
        scheduler = sched_fn(optimizer, **sched_cfg.get("kwargs", {}))

    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience")
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask

    best_val_acc = float("-inf")
    best_test_acc = 0.0
    best_epoch = 0
    best_y_true = None
    best_y_pred = None
    best_y_prob = None
    time_to_best_s = 0.0
    bad_epochs = 0
    epochs_trained = 0

    synchronize_if_cuda(device)
    train_start = time.perf_counter()

    iterator = range(1, epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc="train")

    for epoch in iterator:
        epochs_trained = epoch
        model.train()
        optimizer.zero_grad()

        out = model(graph)
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        with torch.no_grad():
            train_acc = (
                (out[train_mask].argmax(1) == graph.y[train_mask]).float().mean().item()
            )

        model.eval()
        with torch.no_grad():
            eval_out = model(graph)
            val_loss = criterion(eval_out[val_mask], graph.y[val_mask]).item()
            test_loss = criterion(eval_out[test_mask], graph.y[test_mask]).item()
            val_acc = (
                (eval_out[val_mask].argmax(1) == graph.y[val_mask])
                .float()
                .mean()
                .item()
            )
            test_acc = (
                (eval_out[test_mask].argmax(1) == graph.y[test_mask])
                .float()
                .mean()
                .item()
            )

        if logger is not None:
            logger.add_scalar("loss/train", loss.item(), epoch)
            logger.add_scalar("loss/val", val_loss, epoch)
            logger.add_scalar("loss/test", test_loss, epoch)
            logger.add_scalar("acc/train", train_acc, epoch)
            logger.add_scalar("acc/val", val_acc, epoch)
            logger.add_scalar("acc/test", test_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            synchronize_if_cuda(device)
            time_to_best_s = time.perf_counter() - train_start

            best_prob = torch.softmax(eval_out[test_mask], dim=1)
            best_pred = best_prob.argmax(dim=1)

            best_y_true = graph.y[test_mask].detach().cpu().numpy()
            best_y_pred = best_pred.detach().cpu().numpy()
            best_y_prob = best_prob.detach().cpu().numpy()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            iterator.set_description(
                f"loss={loss.item():.4f} tr={train_acc:.3f} va={val_acc:.3f} te={test_acc:.3f}"
            )
        if patience is not None and bad_epochs >= patience:
            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, patience {patience})"
                )
            break

    synchronize_if_cuda(device)
    train_time_s = time.perf_counter() - train_start
    elapsed_time_s = time.perf_counter() - total_start

    if logger is not None:
        logger.add_scalar("final/best_val_acc", best_val_acc, 0)
        logger.add_scalar("final/best_test_acc", best_test_acc, 0)
        logger.add_scalar("final/train_time_s", train_time_s, 0)
        logger.add_scalar("final/elapsed_time_s", elapsed_time_s, 0)
        logger.add_scalar("final/time_to_best_s", time_to_best_s, 0)
        logger.close()

    if verbose:
        print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Test acc at best val: {best_test_acc:.4f}")
        print(f"Timing: train={train_time_s:.2f}s elapsed={elapsed_time_s:.2f}s "
              f"best@={time_to_best_s:.2f}s")

    return {
        "seed": seed,
        "data_path": data_path,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "train_time_s": train_time_s,
        "eval_time_s": 0.0,
        "elapsed_time_s": elapsed_time_s,
        "time_to_best_s": time_to_best_s,
        "y_true": best_y_true,
        "y_pred": best_y_pred,
        "y_prob": best_y_prob,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    run_once(
        config,
        seed=config["seed"],
        data_path=config["data"]["path"],
        logdir=os.path.join(config["logging"]["path"], config["logging"]["experiment"]),
        verbose=True,
    )


if __name__ == "__main__":
    main()
