"""Lightning module for object condensation training."""

import collections
from typing import Any

from torch import Tensor
from torch_geometric.data import Data

from gnn_tracking.metrics.losses import BackgroundLoss, PotentialLoss
from gnn_tracking.postprocessing.dbscanscanner import DBSCANHyperParamScanner
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class TCModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        potential_loss: PotentialLoss = PotentialLoss(),  # noqa: B008
        background_loss: BackgroundLoss = BackgroundLoss(),  # noqa: B008
        cluster_scanner: DBSCANHyperParamScanner | None = None,
        lw_repulsive: float = 1.0,
        lw_background: float = 1.0,
        **kwargs,
    ):
        """Object condensation for tracks. This lightning module implements
        losses, training, and validation steps. k:w


        Args:
            potential_loss:
            background_loss:
            cluster_scanner:
            lw_repulsive: Loss weight for repulsive part of potential loss
            lw_background: Loss weight for background loss
            **kwargs: Passed on to `TrackingModule`
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(
            "lw_repulsive",
            "lw_background",
        )
        self.potential_loss = obj_from_or_to_hparams(
            self, "potential_loss", potential_loss
        )
        self.background_loss = obj_from_or_to_hparams(
            self, "background_loss", background_loss
        )
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
        losses = self.potential_loss(
            x=out["H"],
            particle_id=data.particle_id,
            beta=out["B"],
            pt=data.pt,
            reconstructable=data.reconstructable,
            ec_hit_mask=out.get("ec_hit_mask", None),
        )
        losses["background"] = self.background_loss(
            beta=out["B"],
            particle_id=data.particle_id,
            ec_hit_mask=out.get("ec_hit_mask", None),
        )
        lws = {
            "attractive": 1.0,
            "repulsive": self.hparams["lw_repulsive"],
            "background": self.hparams["lw_background"],
        }
        loss = sum(lws[k] * v for k, v in losses.items())
        losses |= {f"{k}_weighted": v * lws[k] for k, v in losses.items()}
        losses["total"] = loss
        return loss, to_floats(losses)

    @tolerate_some_oom_errors
    def training_step(self, data: Data, batch_idx: int) -> Tensor:
        data = self.data_preproc(data)
        out = self(data)
        loss, loss_dct = self.get_losses(out, data)
        self.log_dict(loss_dct, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> None:
        data = self.data_preproc(data)
        out = self(data)
        loss, metrics = self.get_losses(out, data)
        metrics |= self._evaluate_cluster_metrics(out, data, batch_idx)
        self.log_dict_with_errors(
            metrics, batch_size=self.trainer.val_dataloaders.batch_size
        )

    def _evaluate_cluster_metrics(
        self, out: dict[str, Any], data: Data, batch_idx: int
    ) -> dict[str, float]:
        """Evaluate cluster metrics.

        The cluster metrics need to be evaluated all at once, so we save the
        required inputs to CPU memory until the last validation batch and
        then evaluate the metrics.
        """
        if self.cluster_scanner is None:
            return {}
        self._save_cluster_input(out, data)
        if not self.is_last_val_batch(batch_idx):
            return {}
        cluster_result = self.cluster_scanner(
            **self._cluster_scan_input,
            epoch=self.current_epoch,
            start_params=self._best_cluster_params,
        )
        metrics = cluster_result.metrics
        self._best_cluster_params = cluster_result.best_params
        # todo: Generalize for multiple cluster scanners
        metrics |= {
            f"best_dbscan_{param}": val
            for param, val in cluster_result.best_params.items()
        }
        return metrics

    def _save_cluster_input(self, out: dict[str, Any], data: Data):
        """Save inputs for cluster analysis."""
        inpt = {
            "data": out["H"],
            "truth": data.particle_id,
            "sectors": data.sector,
            "pts": data.pt,
            "reconstructable": data.reconstructable,
            "node_mask": out.get("ec_hit_mask", None),
        }
        for key, value in inpt.items():
            if isinstance(value, Tensor):
                value = value.detach().cpu().numpy()
            self._cluster_scan_input[key].append(value)

    def highlight_metric(self, metric: str) -> bool:
        return metric in [
            "attractive",
            "repulsive",
            "trk.lhc_pt0.9",
            "trk.perfect_pt0.9",
            "trk.double_majority_pt0.9",
        ]
