import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from dhn.augmentation_utils import (
    RESOURCE_METRIC_KEYS,
    atomic_torch_save,
    atomic_write_csv,
    cuda_memory_stats,
    flat_resource_metrics,
    process_peak_rss_bytes,
    reset_cuda_peak,
    resolve_device,
    set_determinism,
)
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Result/checkpoint directory. Default: "
            "results/original_graph_classification/<dataset>."
        ),
    )
    parser.add_argument(
        "--device",
        default="",
        help="Override config device, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    set_determinism(seed)


def synchronize_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


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

    runtime_device = resolve_device(args.device or config["device"])
    device = str(runtime_device)
    dataset_name = config["data"]["dataset"]
    output_dir = args.out_dir or (
        Path("results")
        / "original_graph_classification"
        / dataset_name.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logdir = os.path.join(config["logging"]["path"], config["logging"]["experiment"])
    logger = SummaryWriter(log_dir=logdir)
    log_step = 0

    dataset = HomDataset(
        name=config["data"]["dataset"],
        root_path=config["data"]["root_path"],
    )

    indices = build_splits(config, dataset)

    fold_rows = []
    for fold, (tr_indices, val_indices) in enumerate(indices):
        fold_start = time.perf_counter()
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

        reset_cuda_peak(runtime_device)
        synchronize_if_cuda(runtime_device)
        train_start = time.perf_counter()
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
        synchronize_if_cuda(runtime_device)
        train_time_s = time.perf_counter() - train_start
        training_gpu = cuda_memory_stats(runtime_device)

        checkpoint_path = output_dir / (
            f"{dataset_name}_seed{config['seed']}_fold{fold}.pt"
        )
        atomic_torch_save(
            {
                "model": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "dataset": dataset_name,
                "seed": int(config["seed"]),
                "fold": int(fold),
                "config": config,
            },
            checkpoint_path,
        )

        reset_cuda_peak(runtime_device)
        synchronize_if_cuda(runtime_device)
        eval_start = time.perf_counter()
        final_accuracy = evaluate(
            model,
            dataloader=val_loader,
            logger=logger,
            log_step=log_step,
            fold=fold,
            device=device,
        )
        synchronize_if_cuda(runtime_device)
        eval_time_s = time.perf_counter() - eval_start
        inference_gpu = cuda_memory_stats(runtime_device)
        resources = flat_resource_metrics(
            model,
            training_gpu=training_gpu,
            inference_gpu=inference_gpu,
            checkpoint_path=checkpoint_path,
            peak_rss_bytes=process_peak_rss_bytes(),
        )
        fold_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(config["seed"]),
                "fold": int(fold),
                "accuracy": float(final_accuracy),
                "train_time_s": float(train_time_s),
                "eval_time_s": float(eval_time_s),
                "elapsed_time_s": float(time.perf_counter() - fold_start),
                **resources,
            }
        )
        print(
            f"fold={fold} accuracy={final_accuracy:.4f} "
            f"train_peak_allocated="
            f"{resources['training_gpu_peak_allocated_bytes']} bytes "
            f"inference_peak_allocated="
            f"{resources['inference_gpu_peak_allocated_bytes']} bytes "
            f"process_peak_rss={resources['process_peak_rss_bytes']} bytes"
        )

    logger.close()
    raw_frame = pd.DataFrame(fold_rows)
    raw_path = output_dir / "graph_classification_raw.csv"
    atomic_write_csv(raw_frame, raw_path)

    numeric_fields = [
        "accuracy",
        "train_time_s",
        "eval_time_s",
        "elapsed_time_s",
        *RESOURCE_METRIC_KEYS,
    ]
    summary_rows = []
    for metric in numeric_fields:
        values = raw_frame[metric].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(config["seed"]),
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n_folds": int(len(values)),
            }
        )
    summary_path = output_dir / "graph_classification_summary.csv"
    atomic_write_csv(pd.DataFrame(summary_rows), summary_path)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
