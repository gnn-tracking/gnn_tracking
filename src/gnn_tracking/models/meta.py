"""Wrappers and other "meta" models."""

import torch.nn
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch_geometric.data import Data

from gnn_tracking.utils.lightning import obj_from_or_to_hparams


class Sequential(torch.nn.Module, HyperparametersMixin):
    def __init__(self, layers: list[torch.nn.Module]):
        """Sequentially apply modules for and take care of hyperparameters."""
        super().__init__()
        self._layers = [
            obj_from_or_to_hparams(self, f"squential_layer_{i}", m)
            for i, m in enumerate(layers)
        ]
        self.hparams["layers"] = [
            self.hparams.pop(f"sequential_layer_{i}") for i in range(len(layers))
        ]

    def forward(self, data: Data) -> Data:
        """Forward."""
        for layer in self._layers:
            data = layer(data)
        return data
