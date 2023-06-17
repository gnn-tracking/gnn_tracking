from typing import Any

from torch import Tensor
from torch import Tensor as T
from torch_geometric.data import Data

from gnn_tracking.metrics.losses import GraphConstructionHingeEmbeddingLoss
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import to_floats
from gnn_tracking.utils.lightning import save_sub_hyperparameters


class MLModule(TrackingModule):
    def __init__(
        self,
        loss_fct: GraphConstructionHingeEmbeddingLoss,
        lw_repulsive=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        save_sub_hyperparameters(self, "loss_fct", loss_fct)
        self.loss_fct = loss_fct

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
        return loss, to_floats(loss_dct)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        out = self.forward(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(loss_dct, prog_bar=True, on_step=True)
        self.log("total", loss.float(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, bach_idx: int):
        out = self.forward(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(to_floats(loss_dct), on_epoch=True)
        self.log("total", loss.float(), on_epoch=True)
        # todo: add graph analysis

    def test_step(self, batch, batch_idx: int):
        self.validation_step(batch, batch_idx)
