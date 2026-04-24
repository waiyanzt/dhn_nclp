import torch.nn as nn
import torch.optim as optim
import torch
from .layers import HomConv


def build_layer(layer_config, act_module=nn.ReLU, prev_out=None, **kwargs):
    layer = torch.nn.ModuleList()
    out_dim = 0
    for kernel_name, (indim, outdim, kernel_size) in layer_config.items():
        if prev_out is not None and indim == -1:
            indim = prev_out
        layer.append(HomConv(indim, outdim, act_module, kernel_size, kernel_name, **kwargs))
        out_dim += outdim
    return layer, out_dim


def get_act_module(name):
    return getattr(nn, name)


def get_lr_scheduler(name):
    return getattr(torch.optim.lr_scheduler, name)


def get_optimizer(name):
    return getattr(torch.optim, name)


def get_criterion(name):
    return getattr(nn, name)