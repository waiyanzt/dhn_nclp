import torch
import torch.nn as nn


class HomConv(torch.nn.Module):
    def __init__(self, 
        in_dim, 
        out_dim, 
        act_module=nn.ReLU,
        kernel_size=6, 
        kernel_name=None, 
        p=0.05,
        **kwargs):
        """
        Fast homomorphism based on precompute mappings
        """
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.p = p
        if kernel_name is None:
            self.kernel_name = f'c{kernel_size}'
        else:
            self.kernel_name = kernel_name
        self.f = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    act_module(**kwargs),
                    nn.Dropout(p=self.p),
                    nn.Linear(out_dim, out_dim)
                ) for _ in range(kernel_size)
            ]
        )

    def forward(self, x, mapping_index):
        """
        x: node features
        mapping_index: (num_hom, hom_size)
        """
        # Skip layer if no mapping
        if mapping_index is None or mapping_index[0] is None:
            return self.f[0](x)

        product = 1
        for i, f in enumerate(self.f):
            product *= f(x[mapping_index[:, i]])
        output = torch.zeros(x.size(0), product.size(1)).to(x.device)
        output.scatter_add_(0, mapping_index[:, 0].unsqueeze(1).expand(-1, product.size(1)), product)

        return output

