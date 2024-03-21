import torch
import torch.nn as nn
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch.nn import CrossEntropyLoss

class CEL(nn.Module, HyperparametersMixin):
    def __init__(
            self,
            weight
        ):
        super().__init__()
        self._loss_fct = CrossEntropyLoss(weight=torch.tensor(weight))
        self.weight = weight
        self.save_hyperparameters()

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss_fct(input, target)