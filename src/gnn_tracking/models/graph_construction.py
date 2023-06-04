from __future__ import annotations

import math

import numpy as np
import torch.nn
from torch import Tensor as T
from torch.nn import Linear, ModuleList, init
from torch.nn.functional import normalize, relu
from torch_geometric.data import Data


class GraphConstructionFCNN(torch.nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        beta: float = 0.4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.beta = beta

        self.encoder = Linear(in_dim, hidden_dim, bias=False)
        self.decoder = Linear(hidden_dim, out_dim, bias=False)

        self.layers = ModuleList(
            [Linear(hidden_dim, hidden_dim, bias=False) for _ in range(depth - 1)]
        )
        self.latent_normalization = torch.nn.Parameter(
            torch.Tensor([1.0]), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        self._reset_layer_parameters(self.encoder, var=1 / self.in_dim)
        for layer in self.layers:
            self._reset_layer_parameters(layer, var=2 / self.hidden_dim)
        self._reset_layer_parameters(self.decoder, var=2 / self.hidden_dim)

    @staticmethod
    def _reset_layer_parameters(layer, var: float):
        layer.reset_parameters()
        for p in layer.parameters():
            init.normal_(p.data, mean=0, std=math.sqrt(var))

    def forward(self, data: Data) -> dict[str, T]:
        x = normalize(data.x, p=2.0, dim=1, eps=1e-12, out=None)
        x = self.encoder(x)
        for layer in self.layers:
            x = np.sqrt(self.beta) * layer(relu(x)) + np.sqrt(1 - self.beta) * x
        x = self.decoder(relu(x))
        x *= self.latent_normalization
        return {"H": x}
