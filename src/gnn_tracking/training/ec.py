from typing import Any

from pytorch_lightning.callbacks import RichProgressBar
from torch import Tensor
from torch import Tensor as T
from torch import nn
from torch_geometric.data import Data

from gnn_tracking.metrics.binary_classification import (
    get_maximized_bcs,
    get_roc_auc_scores,
)
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.loading import TrackingDataModule


class ECModule(TrackingModule):
    def __init__(
        self,
        *,
        loss_fct: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_fct = loss_fct

    def get_losses(self, out: dict[str, Any], data: Data) -> T:
        return self.loss_fct(
            x=out["H"],
            particle_id=data.particle_id,
            batch=data.batch,
            edge_index=data.edge_index,
            pt=data.pt,
        )

    def training_step(self, batch, batch_idx: int) -> Tensor | None:
        out = self.model(batch)
        loss = self.get_losses(out, batch)
        self.log("total", loss.float(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, bach_idx: int):
        out = self.model(batch)
        loss = self.get_losses(out, batch)
        self.log("total", loss.float(), on_epoch=True)
        metrics = {}
        # todo: this needs to be done for different pt thresholds
        metrics |= get_roc_auc_scores(
            true=batch.y, predicted=out["w"], max_fprs=[None, 0.01, 0.001]
        )
        metrics |= get_maximized_bcs(y=batch.y, output=out["w"])
        # todo: add graph analysis
        self.loc_dict(metrics, on_epoch=True)


def cli_main():
    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa F841
        ECModule,
        datamodule_class=TrackingDataModule,
        trainer_defaults=dict(callbacks=[RichProgressBar(leave=True)]),
    )


if __name__ == "__main__":
    cli_main()
