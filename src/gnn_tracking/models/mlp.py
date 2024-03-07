"""Fully connected neural network implementations"""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

import math

import numpy as np
import torch
import torch.nn
import torch.nn as nn
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor as T
from torch.nn import Linear, ModuleList, init
from torch.nn.functional import normalize, relu

from gnn_tracking.utils.asserts import assert_feat_dim


class MLP(nn.Module, HyperparametersMixin):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int | None,
        L=3,
        *,
        bias=True,
        include_last_activation=False,
    ):
        """Multi Layer Perceptron, using ReLu as activation function.

        Args:
            input_size: Input feature dimension
            output_size:  Output feature dimension
            hidden_dim: Feature dimension of the hidden layers. If None: Choose maximum
                of input/output size
            L: Total number of layers (1 initial layer, L-2 hidden layers, 1 output
                layer)
            bias: Include bias in linear layer?
            include_last_activation: Include activation function for the last layer?
        """
        super().__init__()
        self.save_hyperparameters()
        if hidden_dim is None:
            hidden_dim = max(input_size, output_size)
        layers = [nn.Linear(input_size, hidden_dim, bias=bias)]
        for _l in range(1, L - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_size, bias=bias))
        if include_last_activation:
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResFCNN(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        alpha: float = 0.6,
        bias: bool = True,
    ):
        """Fully connected NN with residual connections.

        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension = embedding space
            depth: 1 input layer, `depth-1` hidden layers, 1 output layer
            beta: Strength of residual connection in layer-to-layer connections
        """

        super().__init__()
        self.save_hyperparameters()

        self._encoder = Linear(in_dim, hidden_dim, bias=bias)
        self._decoder = Linear(hidden_dim, out_dim, bias=bias)

        self._layers = ModuleList(
            [Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(depth - 1)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        self._reset_layer_parameters(self._encoder, var=1 / self.hparams.in_dim)
        for layer in self._layers:
            self._reset_layer_parameters(layer, var=2 / self.hparams.hidden_dim)
        self._reset_layer_parameters(self._decoder, var=2 / self.hparams.hidden_dim)

    @staticmethod
    def _reset_layer_parameters(layer, var: float):
        layer.reset_parameters()
        for p in layer.parameters():
            init.normal_(p.data, mean=0, std=math.sqrt(var))

    def forward(self, x: T, **ignore) -> T:
        assert_feat_dim(x, self.hparams.in_dim)
        x = normalize(x, p=2.0, dim=1, eps=1e-12, out=None)
        x = self._encoder(x)
        for layer in self._layers:
            x = np.sqrt(self.hparams.alpha) * x + np.sqrt(
                1 - self.hparams.alpha
            ) * layer(relu(x))
        x = self._decoder(relu(x))
        assert x.shape[1] == self.hparams.out_dim
        return x


def get_pixel_mask(layer: T) -> T:
    return torch.isin(layer, torch.tensor(list(range(18))))


class HeterogeneousResFCNN(nn.Module, HyperparametersMixin):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int,
        alpha: float = 0.6,
        bias: bool = True,
    ):
        """Separate FCNNs for pixel and strip data, with residual connections.
        For parameters, see `ResFCNN`.
        """
        super().__init__()
        self.pixel_fcnn = ResFCNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            alpha=alpha,
            bias=bias,
        )
        self.strip_fcnn = ResFCNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            alpha=alpha,
            bias=bias,
        )
        self.save_hyperparameters()

    def forward(self, x: T, layer: T) -> T:
        pixel_mask = get_pixel_mask(layer)
        x_pixel = x[pixel_mask]
        x_strip = x[~pixel_mask]

        embed_pixel = self.pixel_fcnn(x_pixel)
        embed_strip = self.strip_fcnn(x_strip)

        # We can simply concatenate without destroying the
        # existing order, because the data is already sorted
        # by pixel and then strip.
        return torch.vstack([embed_pixel, embed_strip])
