"""Models for filtering out noise before we even build a graph"""

from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor as T
from torch import nn
from torch_geometric.data import Data

from gnn_tracking.utils.lightning import obj_from_or_to_hparams


class TruthNoiseClassifierModel(nn.Module):
    def __init__(self):
        """Remove all noise with truth information"""
        super().__init__()

    def forward(self, data: Data) -> Data:
        return data.subgraph(data.particle_id != 0)


class WithNoiseClassification(nn.Module, HyperparametersMixin):
    def __init__(self, noise_model: nn.Module, model: nn.Module):
        """Combine a noise filter with another model"""
        super().__init__()
        self.noise_model = obj_from_or_to_hparams(self, "noise_model", noise_model)
        self.model = obj_from_or_to_hparams(self, "normal_model", model)

    def forward(self, data: Data) -> dict[str, T | None]:
        mask = self.noise_model(data)
        masked_data = data.subgraph(mask)
        out = self.model(masked_data)
        out["hit_mask"] = mask
        return out
