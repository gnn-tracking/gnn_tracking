import math

import numpy as np
import torch
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor as T
from torch.nn import Linear, ModuleList, init
from torch.nn.functional import normalize, relu, sigmoid
from torch_geometric.data import Data

from gnn_tracking.models.mlp import MLP
from gnn_tracking.training.ec import ECModule

# todo: Need a lightning class for these


class EFDeepSet(ECModule):
    def __init__(
        self,
        *,
        in_dim: int = 14,
        hidden_dim: int = 128,
        depth: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["loss_fct", "optimizer", "scheduler"])

        self.node_encoder = MLP(
            input_size=in_dim,
            output_size=hidden_dim,
            hidden_dim=hidden_dim,
            L=depth,
            bias=False,
            include_last_activation=True,
        )
        self.aggregator = MLP(
            input_size=2 * hidden_dim,
            output_size=1,
            L=depth,
            hidden_dim=2 * hidden_dim,
            bias=False,
        )
        # self.reset_parameters()

    def forward(self, data: Data) -> dict[str, T]:
        x = normalize(data.x, p=2.0, dim=1, eps=1e-12, out=None)
        x_encoded = self.node_encoder(x)
        i = data.edge_index[0]
        j = data.edge_index[1]
        xi = x_encoded[i]
        xj = x_encoded[j]
        invariant_1 = torch.abs(xi - xj)
        invariant_2 = xi + xj
        invariant = torch.cat((invariant_1, invariant_2), dim=1)
        epsilon = 1e-8
        w = epsilon + (1 - 2 * epsilon) * sigmoid(self.aggregator(invariant)).squeeze()
        return {"W": w}


class EFMLP(ECModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        beta: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder = Linear(in_dim * 2, hidden_dim, bias=False)
        self.decoder = Linear(hidden_dim, 1, bias=False)

        self.layers = ModuleList(
            [Linear(hidden_dim, hidden_dim, bias=False) for _ in range(depth - 1)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        self._reset_layer_parameters(self.encoder, var=1 / (2 * self.hparams.in_dim))
        for layer in self.layers:
            self._reset_layer_parameters(layer, var=2 / self.hparams.hidden_dim)
        self._reset_layer_parameters(self.decoder, var=2 / self.hparams.hidden_dim)

    @staticmethod
    def _reset_layer_parameters(layer, var: float):
        layer.reset_parameters()
        for p in layer.parameters():
            init.normal_(p.data, mean=0, std=math.sqrt(var))

    def forward(self, data: Data) -> dict[str, T]:
        x = normalize(data.x, p=2.0, dim=1, eps=1e-12, out=None)
        i = data.edge_index[0]
        j = data.edge_index[1]
        x = torch.cat((x[i], x[j]), dim=1)
        x = self.encoder(x)
        for layer in self.layers:
            x = (
                np.sqrt(self.hparams.beta) * layer(relu(x))
                + np.sqrt(1 - self.hparams.beta) * x
            )
        x = 0.001 + 0.998 * sigmoid(self.decoder(relu(x))).squeeze()
        return {"W": x}


class GeometricEF(torch.nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocals
    def __init__(self, phi_slope_max, z0_max, dR_max):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, data: Data):
        r = data.x[:, 0]
        phi = data.x[:, 1]
        z = data.x[:, 2]
        eta = data.x[:, 3]
        i = data.edge_index[0]
        j = data.edge_index[1]
        dz = z[i] - z[j]
        dr = r[i] - r[j]
        dphi = phi[i] - phi[j]
        deta = eta[i] - eta[j]
        dR = torch.sqrt(deta**2 + dphi**2)
        phi_slope = dphi / dR
        z0 = z[i] - r[i] * dz / dr
        return (
            (phi_slope.abs() < self.hparams.phi_slope_max)
            & (z0.abs() < self.hparams.z0_max)
            & (dR.abs() < self.hparams.dR_max)
        )
