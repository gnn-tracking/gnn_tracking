from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dim: int, L=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        for _l in range(1, L - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_size))
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
