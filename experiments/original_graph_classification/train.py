import os
import argparse

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from dhn.models import DHN
from dhn.datasets import HomDataLoader, HomDataset
from dhn.utils import (
    get_act_module,
    get_criterion,
    get_optimizer,
    get_lr_scheduler,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DHN model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training config file",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    logger,
    log_step,
    scheduler=None,
    fold=None,
    device="cpu",
):
    model.train()
    local_log_step = 0
    for gdata in dataloader:
        gdata = gdata.to(device)
        optimizer.zero_grad()
        outputs = model(gdata)
        loss = criterion(outputs, gdata.y)
        loss.backward()
        optimizer.step()
        logger.add_scalar(f"loss/train/{fold}", loss.item(), log_step + local_log_step)
        local_log_step += 1
    if scheduler:
        scheduler.step()
    return log_step + local_log_step


@torch.no_grad()
def evaluate(model, dataloader, logger, log_step, fold=None, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    for gdata in dataloader:
        gdata = gdata.to(device)
        outputs = model(gdata)
        predicted = outputs.argmax(1)
        correct += (predicted == gdata.y).sum().item()
        total += len(gdata.y)
    accuracy = correct / total if total > 0 else 0.0
    logger.add_scalar(f"acc/val/{fold}", accuracy, log_step)
    return accuracy


def build_splits(config, dataset):
    if config["data"]["cross_validation"]:
        labels = [dataset[i].y.item() for i in range(len(dataset))]
        kfold = StratifiedKFold(
            n_splits=10,
            random_state=config["seed"],
            shuffle=True,
        )
        return list(kfold.split(labels, labels))

    train_path = os.path.join(
        config["data"]["root_path"], config["data"]["train_data_path"]
    )
    val_path = os.path.join(
        config["data"]["root_path"], config["data"]["val_data_path"]
    )
    tri = np.fromfile(train_path, sep=" ").astype(int)
    vai = np.fromfile(val_path, sep=" ").astype(int)
    return [(tri, vai)]


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = config["device"]
    logdir = os.path.join(config["logging"]["path"], config["logging"]["experiment"])
    logger = SummaryWriter(log_dir=logdir)
    log_step = 0

    dataset = HomDataset(
        name=config["data"]["dataset"],
        root_path=config["data"]["root_path"],
    )

    indices = build_splits(config, dataset)

    for fold, (tr_indices, val_indices) in enumerate(indices):
        train_loader = HomDataLoader(
            [dataset[int(i)] for i in tr_indices],
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )
        val_loader = HomDataLoader(
            [dataset[int(i)] for i in val_indices],
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )

        model = DHN(
            out_dim=config["model"]["out_dim"],
            layers_config=config["model"]["layers_config"],
            act_module=get_act_module(config["model"]["activation"]["name"]),
            agg=config["model"]["agg"],
            **config["model"]["activation"]["kwargs"],
        ).to(device)

        criterion_fn = get_criterion(config["training"]["loss"]["name"])
        criterion = criterion_fn(**config["training"]["loss"]["kwargs"])

        optimizer_fn = get_optimizer(config["training"]["optimizer"]["name"])
        optimizer = optimizer_fn(
            params=model.parameters(),
            **config["training"]["optimizer"]["kwargs"],
        )

        scheduler = None
        if config["training"]["lr_scheduling"]["name"]:
            scheduler_fn = get_lr_scheduler(config["training"]["lr_scheduling"]["name"])
            scheduler = scheduler_fn(
                optimizer, **config["training"]["lr_scheduling"]["kwargs"]
            )

        for _ in tqdm(range(1, config["training"]["epochs"] + 1), desc=f"fold {fold}"):
            log_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                logger=logger,
                log_step=log_step,
                scheduler=scheduler,
                fold=fold,
                device=device,
            )
            evaluate(
                model,
                dataloader=val_loader,
                logger=logger,
                log_step=log_step,
                fold=fold,
                device=device,
            )

    logger.close()


if __name__ == "__main__":
    main()
