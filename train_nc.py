"""Train DHN for node classification on a precomputed graph bundle."""
import os
import argparse

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dhn.models import DHN
from dhn.datasets import NodeClassDataset
from dhn.utils import (
    get_act_module,
    get_criterion,
    get_optimizer,
    get_lr_scheduler,
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


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['seed'])

    device = config['device']
    if 'cuda' in device and not torch.cuda.is_available():
        print(f"WARNING: requested {device} but CUDA unavailable; falling back to CPU")
        device = 'cpu'

    logdir = os.path.join(config['logging']['path'], config['logging']['experiment'])
    logger = SummaryWriter(log_dir=logdir)
    print(f"Logging to {logdir}")

    ds = NodeClassDataset(config['data']['path'])
    graph = move_graph_to(ds.data, device)
    feat_dim = ds.num_features
    num_classes = ds.num_classes
    print(f"Loaded data: {graph.num_nodes} nodes, {feat_dim} features, {num_classes} classes")
    print(f"Splits: train={int(graph.train_mask.sum())} "
          f"val={int(graph.val_mask.sum())} test={int(graph.test_mask.sum())}")

    layers_config = resolve_layers_config(config['model']['layers_config'], feat_dim)
    activation_kwargs = config['model']['activation'].get('kwargs', {})
    model = DHN(
        out_dim=num_classes,
        layers_config=layers_config,
        act_module=get_act_module(config['model']['activation']['name']),
        agg=config['model']['agg'],
        **activation_kwargs,
    ).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params}")

    criterion_fn = get_criterion(config['training']['loss']['name'])
    criterion = criterion_fn(**config['training']['loss'].get('kwargs', {}))

    optimizer_fn = get_optimizer(config['training']['optimizer']['name'])
    optimizer = optimizer_fn(
        model.parameters(),
        **config['training']['optimizer'].get('kwargs', {}),
    )

    scheduler = None
    sched_cfg = config['training'].get('lr_scheduling', {})
    if sched_cfg.get('name'):
        sched_fn = get_lr_scheduler(sched_cfg['name'])
        scheduler = sched_fn(optimizer, **sched_cfg.get('kwargs', {}))

    epochs = config['training']['epochs']
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    pbar = tqdm(range(1, epochs + 1), desc='train')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        with torch.no_grad():
            train_acc = (out[train_mask].argmax(1) == graph.y[train_mask]).float().mean().item()
        val_acc, val_loss = evaluate(model, graph, val_mask, criterion)
        test_acc, test_loss = evaluate(model, graph, test_mask, criterion)

        logger.add_scalar('loss/train', loss.item(), epoch)
        logger.add_scalar('loss/val', val_loss, epoch)
        logger.add_scalar('loss/test', test_loss, epoch)
        logger.add_scalar('acc/train', train_acc, epoch)
        logger.add_scalar('acc/val', val_acc, epoch)
        logger.add_scalar('acc/test', test_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

        pbar.set_description(
            f"loss={loss.item():.4f} tr={train_acc:.3f} va={val_acc:.3f} te={test_acc:.3f}"
        )

    print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Test acc at best val: {best_test_acc:.4f}")
    logger.add_scalar('final/best_val_acc', best_val_acc, 0)
    logger.add_scalar('final/best_test_acc', best_test_acc, 0)
    logger.close()


if __name__ == '__main__':
    main()
