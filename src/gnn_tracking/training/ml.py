"""Pytorch lightning module with training and validation step for the metric learning
approach to graph construction.
"""

from typing import Any

from torch import Tensor
from torch import Tensor as T
from torch_geometric.data import Data

from gnn_tracking.metrics.losses import GraphConstructionHingeEmbeddingLoss
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class MLModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        loss_fct: GraphConstructionHingeEmbeddingLoss,
        lw_repulsive=1.0,
        **kwargs,
    ):
        """Pytorch lightning module with training and validation step for the metric
        learning approach to graph construction.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters("lw_repulsive")
        self.loss_fct = obj_from_or_to_hparams(self, "loss_fct", loss_fct)

    # noinspection PyUnusedLocal
    def get_losses(self, out: dict[str, Any], data: Data) -> tuple[T, dict[str, float]]:
        loss_dct = self.loss_fct(
            x=out["H"],
            particle_id=data.particle_id,
            batch=data.batch,
            edge_index=data.edge_index,
            pt=data.pt,
        )
        lws = {
            "attractive": 1.0,
            "repulsive": self.hparams["lw_repulsive"],
        }
        loss = sum(lws[k] * v for k, v in loss_dct.items())
        loss_dct |= {f"{k}_weighted": v * lws[k] for k, v in loss_dct.items()}
        loss_dct["total"] = loss
        return loss, to_floats(loss_dct)

    @tolerate_some_oom_errors
    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        batch = self.data_preproc(batch)
        out = self(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(loss_dct, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        batch = self.data_preproc(batch)
        out = self(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict_with_errors(
            loss_dct, batch_size=self.trainer.val_dataloaders.batch_size
        )
        # todo: add graph analysis
