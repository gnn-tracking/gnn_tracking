"""Lightning module for object condensation training."""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

import collections
from typing import Any

from torch import Tensor
from torch_geometric.data import Data

from gnn_tracking.metrics.losses import MultiLossFct
from gnn_tracking.postprocessing.clusterscanner import ClusterScanner
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import add_key_suffix, to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class TCModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        loss_fct: MultiLossFct,
        cluster_scanner: ClusterScanner | None = None,
        **kwargs,
    ):
        """Object condensation for tracks. This lightning module implements
        losses, training, and validation steps. k:w


        Args:
            loss_fct:
            cluster_scanner:
            **kwargs: Passed on to `TrackingModule`
        """
        super().__init__(**kwargs)
        self.loss_fct = obj_from_or_to_hparams(self, "loss_fct", loss_fct)
        self.cluster_scanner = obj_from_or_to_hparams(
            self, "cluster_scanner", cluster_scanner
        )
        self._cluster_scan_input = collections.defaultdict(list)
        self._best_cluster_params = {}

    def is_last_val_batch(self, batch_idx: int) -> bool:
        """Are we validating the last batch of the validation set?"""
        return batch_idx == self.trainer.num_val_batches[0] - 1

    def get_losses(
        self, out: dict[str, Any], data: Data
    ) -> tuple[Tensor, dict[str, float]]:
        losses = self.loss_fct(
            x=out["H"],
            particle_id=data.particle_id,
            beta=out["B"],
            pt=data.pt,
            reconstructable=data.reconstructable,
            eta=data.eta,
            ec_hit_mask=out.get("ec_hit_mask"),
            batch=data.batch,
            true_edge_index=getattr(data, "true_edges", None),
        )
        metrics = (
            losses.loss_dct
            | to_floats(add_key_suffix(losses.weighted_losses, "_weighted"))
            | to_floats(losses.extra_metrics)
        )
        metrics["total"] = float(losses.loss)
        return losses.loss, metrics

    @tolerate_some_oom_errors
    def training_step(self, data: Data, batch_idx: int) -> Tensor:
        data = self.data_preproc(data)
        out = self(data, _preprocessed=True)
        loss, loss_dct = self.get_losses(out, data)
        self.log_dict(
            add_key_suffix(loss_dct, "_train"),
            prog_bar=True,
            on_step=True,
            batch_size=self.trainer.train_dataloader.batch_size,  # pyright: ignore[reportOptionalMemberAccess]
        )
        assert isinstance(loss, Tensor)
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> None:
        data = self.data_preproc(data)
        out = self(data, _preprocessed=True)
        loss, metrics = self.get_losses(out, data)
        metrics |= self._evaluate_cluster_metrics(out, data, batch_idx)
        self.log_dict_with_errors(
            metrics,
            batch_size=self.trainer.val_dataloaders.batch_size,  # pyright: ignore[reportOptionalMemberAccess]
        )

    def _evaluate_cluster_metrics(
        self, out: dict[str, Any], data: Data, batch_idx: int
    ) -> dict[str, float]:
        """Evaluate cluster metrics."""
        if self.cluster_scanner is None:
            return {}
        self.cluster_scanner(data, out, batch_idx)
        if not self.is_last_val_batch(batch_idx):
            return {}
        return self.cluster_scanner.get_foms()

    def highlight_metric(self, metric: str) -> bool:
        return metric in [
            "attractive",
            "repulsive",
            "trk.lhc_pt0.9",
            "trk.perfect_pt0.9",
            "trk.double_majority_pt0.9",
        ]
