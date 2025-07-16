"""Fully connected neural network implementations"""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

import math
import os
from math import sqrt

import numpy as np
import torch
import torch._dynamo
import torch.nn
import torch.nn as nn
from torch import Tensor as T
from torch.nn import Linear, Module, ModuleList, Tanh, init
from torch.nn.functional import normalize, relu

torch._dynamo.config.suppress_errors = True


class MLP(nn.Module):
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
        if hidden_dim is None:
            hidden_dim = max(input_size, output_size)
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_dim, bias=bias)]
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


class ResFCNN(nn.Module):
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
            depth: 1 input encoder layer, `depth-1` hidden layers, 1 output encoder layer
            alpha: strength of the residual connection
        """
        # WARNING: Do not save_hyperparameters_here because of
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19596

        super().__init__()

        if depth < 1:
            msg = "Depth must be at least 1"
            raise ValueError(msg)

        self._encoder = Linear(in_dim, hidden_dim, bias=bias)
        self._decoder = Linear(hidden_dim, out_dim, bias=bias)

        self._layers = ModuleList(
            [Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(depth - 1)]
        )

        self._reset_layer_parameters(self._encoder, var=1 / in_dim)
        for layer in self._layers:
            self._reset_layer_parameters(layer, var=2 / hidden_dim)
        self._reset_layer_parameters(self._decoder, var=2 / hidden_dim)

        self._alpha = alpha

    @staticmethod
    def _reset_layer_parameters(layer, var: float):
        layer.reset_parameters()
        for p in layer.parameters():
            init.normal_(p.data, mean=0, std=math.sqrt(var))

    def forward(self, x: T, **ignore) -> T:
        x = normalize(x, p=2.0, dim=1, eps=1e-12, out=None)
        x = self._encoder(x)
        for layer in self._layers:
            x = np.sqrt(self._alpha) * x + np.sqrt(1 - self._alpha) * layer(relu(x))
        return self._decoder(relu(x))


def get_pixel_mask(layer: T) -> T:
    return torch.isin(layer, torch.tensor(list(range(18)), device=layer.device))


class HeterogeneousResFCNN(nn.Module):
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
        # WARNING: Do not save_hyperparameters_here because of
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19596
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

    def forward(self, x: T, layer: T) -> T:
        pixel_mask = get_pixel_mask(layer)
        if "PYTEST_CURRENT_TEST" not in os.environ and (
            pixel_mask.all() or not pixel_mask.any()
        ):
            msg = "All or no pixel data found; this doesn't make sense with heterogeneous model"
            raise ValueError(msg)

        x_pixel = x[pixel_mask]
        x_strip = x[~pixel_mask]

        embed_pixel = self.pixel_fcnn(x_pixel)
        embed_strip = self.strip_fcnn(x_strip)

        # We can simply concatenate without destroying the
        # existing order, because the data is already sorted
        # by pixel and then strip.
        return torch.vstack([embed_pixel, embed_strip])


class ResMLP(nn.Module):
    """Fully connected NN w/ residual connections and Gaussian init
    Args:
    in_dim: input dimension
    out_dim: output dimension
    width: # neurons per internal layer
    beta: strength of the residual connection
    gamma_0: tuning of final layer output normalisation
    depth: number of hidden layers
    """

    # hidden
    # detph
    # alpha
    # bias

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int = 4,
        beta: float = 1.0,
        gamma_0: float = 1.0,
        eta_0: float = 0.01,
        activation: Module = Tanh,
        optimizer: str = "adam",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.layers = ModuleList()
        for layer in range(depth + 1):
            self.layers.append(
                Linear(
                    in_dim if (layer == 0) else hidden_dim,
                    out_dim if (layer == depth) else hidden_dim,
                    bias=False,
                )
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = hidden_dim

        self.beta = beta
        self.gamma_0 = gamma_0
        self.eta_0 = eta_0
        self.gamma = gamma_0 * sqrt(hidden_dim)
        self.depth = depth
        self.act = activation()

        self.lr = self.get_lr(optimizer)

        self.reset_parameters()

    def reset_parameters(self):
        for _, weights in enumerate(self.layers):
            for p in weights.weight:
                init.normal_(p.data, mean=0, std=1)

    def get_lr(self, optimizer):
        if "sgd" in optimizer.lower():
            return self.eta_0 * self.gamma_0**2 * self.width
        if "adam" in optimizer.lower():
            return self.eta_0 * self.gamma_0 * sqrt(self.width)
        exception_message = f"Cannot locate parametrization for optimizer {optimizer}"
        raise Exception(exception_message)

    def forward(self, x):
        for layer, weights in enumerate(self.layers):
            if layer == 0:
                x = weights(x) / sqrt(self.in_dim)
            elif (layer > 0) and (layer < self.depth):
                x = x + (self.beta / sqrt(self.depth * self.width)) * weights(
                    self.act(x)
                )
            else:
                x = weights(self.act(x)) / (self.width * self.gamma)
        return x
