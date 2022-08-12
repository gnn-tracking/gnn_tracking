from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, L=3):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        for _l in range(1, L - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
