import torch
import torch.nn as nn
from .layers import HomConv
from .utils import build_layer, get_act_module, get_lr_scheduler


class DHN(torch.nn.Module):
    def __init__(
        self,
        out_dim = 10,
        layers_config = [
            {'c2': (10, 5, 2), 'c3': (10, 5, 3), 'c4': (10, 5, 4)},
            {'c2': (15, 5, 2), 'c3': (15, 5, 3), 'c4': (15, 5, 4)}
        ],
        act_module = nn.ReLU,
        agg = None,
        **kwargs
        ):
        """
        Rooted homomorphism network
        """
        super().__init__()
        self.out_dim = out_dim
        self.layers = torch.nn.ModuleList()
        self.layers_config = layers_config
        self.agg = None
        hom_conv_dim = 0
        for config in layers_config:
            layer, hom_conv_dim = build_layer(config, act_module, hom_conv_dim, **kwargs)
            self.layers.append(layer)
        if agg:
            agg_block = nn.ModuleList()
            prev = hom_conv_dim
            for i, l in enumerate(agg):
                agg_block.append(nn.Linear(prev, l))
                if i < len(agg) - 1:
                    agg_block.append(act_module(**kwargs))
                prev = l
            self.agg = torch.nn.Sequential(*agg_block)
            self.out_dim = hom_conv_dim
        else:
            self.fc = torch.nn.Linear(hom_conv_dim, out_dim)
            self.agg = None

    def forward(self, hom_data):
        feat = hom_data.x
        mapping_index_dict = hom_data.mapping_index_dict
        batch_idx = hom_data.batch
        batch_size = hom_data.batch_size
        for l in self.layers:
            feat = torch.cat([k(feat, mapping_index_dict.get(k.kernel_name, None)) for k in l], dim=1)
        if self.agg:
            h = feat
            batch_agg = torch.zeros(
                (batch_size, self.out_dim), 
                dtype=h.dtype, 
                device=h.device
            )
            batch_agg.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, self.out_dim), h)
            h = self.agg(batch_agg)
        else:
            h = self.fc(feat)
        return h
