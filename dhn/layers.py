import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class HomConv(torch.nn.Module):
    def __init__(self, 
        in_dim, 
        out_dim, 
        act_module=nn.ReLU,
        kernel_size=6, 
        kernel_name=None, 
        p=0.05,
        mapping_chunk_size=None,
        checkpoint_chunks=True,
        **kwargs):
        """
        Fast homomorphism based on precompute mappings
        """
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.p = p
        self.mapping_chunk_size = mapping_chunk_size
        self.checkpoint_chunks = checkpoint_chunks
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

    def _mapping_product(self, x, mapping_index):
        product = 1
        for i, transform in enumerate(self.f):
            product = product * transform(x[mapping_index[:, i]])
        return product

    def forward(self, x, mapping_index):
        """
        x: node features
        mapping_index: (num_hom, hom_size)
        """
        # Skip layer if no mapping
        if mapping_index is None or len(mapping_index) == 0:
            return self.f[0](x)

        chunk_size = self.mapping_chunk_size
        if chunk_size is None or len(mapping_index) <= chunk_size:
            product = self._mapping_product(x, mapping_index)
            output = x.new_zeros((x.size(0), self.out_dim))
            output.index_add_(0, mapping_index[:, 0], product)
            return output

        output = x.new_zeros((x.size(0), self.out_dim))
        for mapping_chunk in mapping_index.split(chunk_size):
            if self.training and torch.is_grad_enabled() and self.checkpoint_chunks:
                product = checkpoint(
                    self._mapping_product,
                    x,
                    mapping_chunk,
                    use_reentrant=False,
                )
            else:
                product = self._mapping_product(x, mapping_chunk)
            output.index_add_(0, mapping_chunk[:, 0], product)
        return output
