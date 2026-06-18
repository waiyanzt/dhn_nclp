"""Train DHN for node classification on a precomputed graph bundle."""

import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dhn.datasets import NodeClassDataset
from dhn.models import DHN
from dhn.utils import (
    get_act_module,
    get_criterion,
    get_lr_scheduler,
    get_optimizer,
)


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

    logger = SummaryWriter(log_dir=logdir)

    if verbose:
        print(f"Logging to {logdir}")

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

    model = DHN(
        out_dim=num_classes,
        layers_config=layers_config,
        act_module=get_act_module(config["model"]["activation"]["name"]),
        agg=config["model"]["agg"],
        **activation_kwargs,
    ).to(device)

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
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    best_y_true = None
    best_y_pred = None
    best_y_prob = None

    start_time = time.time()

    iterator = range(1, epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc="train")

    for epoch in iterator:
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

        val_acc, val_loss = evaluate(model, graph, val_mask, criterion)
        test_acc, test_loss = evaluate(model, graph, test_mask, criterion)

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

            model.eval()
            with torch.no_grad():
                best_out = model(graph)
                best_prob = torch.softmax(best_out[test_mask], dim=1)
                best_pred = best_prob.argmax(dim=1)

                best_y_true = graph.y[test_mask].detach().cpu().numpy()
                best_y_pred = best_pred.detach().cpu().numpy()
                best_y_prob = best_prob.detach().cpu().numpy()

        if verbose:
            iterator.set_description(
                f"loss={loss.item():.4f} tr={train_acc:.3f} va={val_acc:.3f} te={test_acc:.3f}"
            )

    train_time_s = time.time() - start_time

    logger.add_scalar("final/best_val_acc", best_val_acc, 0)
    logger.add_scalar("final/best_test_acc", best_test_acc, 0)
    logger.close()

    if verbose:
        print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Test acc at best val: {best_test_acc:.4f}")

    return {
        "seed": seed,
        "data_path": data_path,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "train_time_s": train_time_s,
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
