from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor

try:
    from torch_cluster import knn
except ImportError:
    knn = None


class DynamicEdgeConv(MessagePassing):
    def __init__(
        self, nn: Callable, k: int, aggr: str = "max", num_workers: int = 1, **kwargs
    ):
        super().__init__(aggr=aggr, flow="source_to_target", **kwargs)

        if knn is None:
            raise ImportError("`DynamicEdgeConv` requires `torch-cluster`.")

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()
        self.edge_index = None

    def reset_parameters(self):
        self.nn.reset_parameters()

    def get_edge_index(self):
        return self.edge_index

    def forward(
        self,
        x: Tensor | PairTensor,
        batch: OptTensor | PairTensor | None = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        self.edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(self.edge_index, x=x, size=None), self.edge_index

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn}, k={self.k})"
