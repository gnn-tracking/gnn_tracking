"""Pytorch lightning module with training and validation step for the metric learning
approach to graph construction.
"""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

from typing import Any

from torch import Tensor
from torch import Tensor as T
from torch_geometric.data import Data

from gnn_tracking.graph_construction.k_scanner import GraphConstructionKNNScanner
from gnn_tracking.metrics.losses import MultiLossFct
from gnn_tracking.metrics.losses.metric_learning import (
    GraphConstructionHingeEmbeddingLoss,
)
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import add_key_suffix, to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class MLModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        loss_fct: MultiLossFct,
        gc_scanner: GraphConstructionKNNScanner | None = None,
        **kwargs,
    ):
        """Pytorch lightning module with training and validation step for the metric
        learning approach to graph construction.
        """
        super().__init__(**kwargs)
        self.loss_fct: GraphConstructionHingeEmbeddingLoss = obj_from_or_to_hparams(
            self, "loss_fct", loss_fct
        )
        self.gc_scanner = obj_from_or_to_hparams(self, "gc_scanner", gc_scanner)

    # noinspection PyUnusedLocal
    def get_losses(self, out: dict[str, Any], data: Data) -> tuple[T, dict[str, float]]:
        if not hasattr(data, "true_edge_index"):
            # For the point cloud data, we unfortunately saved the true edges
            # simply as edge_index.
            data.true_edge_index = data.edge_index
        losses = self.loss_fct(
            x=out["H"],
            particle_id=data.particle_id,
            batch=data.batch,
            true_edge_index=data.true_edge_index,
            pt=data.pt,
            eta=data.eta,
            reconstructable=data.reconstructable,
        )
        metrics = (
            losses.loss_dct
            | to_floats(add_key_suffix(losses.weighted_losses, "_weighted"))
            | losses.extra_metrics
        )
        metrics["total"] = float(losses.loss)
        return losses.loss, metrics

    @tolerate_some_oom_errors
    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        batch = self.data_preproc(batch)
        out = self(batch, _preprocessed=True)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(
            add_key_suffix(loss_dct, "_train"),
            prog_bar=True,
            on_step=True,
            batch_size=self.trainer.train_dataloader.batch_size,
        )
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        batch = self.data_preproc(batch)
        out = self(batch, _preprocessed=True)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict_with_errors(
            loss_dct, batch_size=self.trainer.val_dataloaders.batch_size
        )
        if self.gc_scanner is not None:
            self.gc_scanner(batch, batch_idx, latent=out["H"])

    def on_validation_epoch_end(self) -> None:
        if self.gc_scanner is not None:
            self.log_dict(
                self.gc_scanner.get_foms(),
                on_step=False,
                on_epoch=True,
                batch_size=self.trainer.val_dataloaders.batch_size,
            )

    def highlight_metric(self, metric: str) -> bool:
        return metric in [
            "n_edges_frac_segment50_95",
            "total",
            "attractive",
            "repulsive",
            "max_frac_segment50",
        ]
