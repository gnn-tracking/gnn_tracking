"""Lightning module for edge classifier training."""

from typing import Any

from torch import Tensor
from torch import Tensor as T
from torch import nn
from torch_geometric.data import Data

from gnn_tracking.metrics.binary_classification import (
    get_maximized_bcs,
    get_roc_auc_scores,
)
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.nomenclature import denote_pt
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class ECModule(TrackingModule):
    def __init__(
        self,
        *,
        loss_fct: nn.Module,
        **kwargs,
    ):
        """Lightning module for edge classifier training."""
        super().__init__(**kwargs)
        self.loss_fct = obj_from_or_to_hparams(self, "loss_fct", loss_fct)

    def get_losses(self, out: dict[str, Any], data: Data) -> T:
        return self.loss_fct(
            w=out["W"],
            y=data.y.float(),
            pt=data.pt,
            edge_index=data.edge_index,
        )

    @tolerate_some_oom_errors
    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        batch = self.data_preproc(batch)
        out = self(batch)
        loss = self.get_losses(out, batch)
        self.log("total_train", loss.float(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        batch = self.data_preproc(batch)
        out = self(batch)
        loss = self.get_losses(out, batch)
        self.log("total", loss.float(), on_epoch=True)
        metrics = {}
        for pt in [0.0, 0.5, 0.9, 1.5]:
            if pt > 0:
                pt_mask = (batch.pt[batch.edge_index[0]] > pt) | (
                    batch.pt[batch.edge_index[1]] > pt
                )
                w = out["W"][pt_mask]
                y = batch.y[pt_mask]
            else:
                w = out["W"]
                y = batch.y
            _metrics = get_roc_auc_scores(
                true=y, predicted=w, max_fprs=[None, 0.01, 0.001]
            ) | get_maximized_bcs(y=y, output=w)
            metrics |= denote_pt(_metrics, pt)
        # todo: add graph analysis
        self.log_dict_with_errors(
            metrics,
            batch_size=self.trainer.val_dataloaders.batch_size,
        )

    def highlight_metric(self, metric: str) -> bool:
        return "max_mcc" in metric
