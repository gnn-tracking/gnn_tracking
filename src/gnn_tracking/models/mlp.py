from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        L=3,
        *,
        bias=True,
        include_last_activation=False,
    ):
        """Multi Layer Perceptron, using ReLu as activation function.

        Args:
            input_size: Input feature dimension
            output_size:  Output feature dimension
            hidden_dim: Feature dimension of the hidden layers
            L: Total number of layers (1 initial layer, L-2 hidden layers, 1 output
                layer)
            bias: Include bias in linear layer?
            include_last_activation: Include activation function for the last layer?
        """
        super().__init__()
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
