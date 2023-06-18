import collections
from typing import Any

from torch import Tensor
from torch_geometric.data import Data

from gnn_tracking.metrics.losses import BackgroundLoss, PotentialLoss
from gnn_tracking.postprocessing.clusterscanner import ClusterFctType
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams


class TCModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        potential_loss: PotentialLoss,
        background_loss: BackgroundLoss,
        cluster_scanner: ClusterFctType,
        lw_repulsive: float = 1.0,
        lw_background: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(
            ignore=["potential_loss", "background_loss", "cluster_scanner"]
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

    def get_losses(self, out: dict[str, Any], data: Data):
        losses = self.loss_fct(
            x=out["H"],
            particle_id=data.particle_id,
            beta=out["B"],
            pt=data.pt,
            reconstructable=data.reconstructable,
        )
        losses["background"] = self.background_loss(
            beta=out["B"],
            particle_id=data.particle_id,
        )
        lws = {
            "attractive": 1.0,
            "repulsive": self.hparams["lw_repulsive"],
            "background": self.hparams["lw_background"],
        }
        loss = sum(lws[k] * v for k, v in losses.items())
        losses |= {f"{k}_weighted": v * lws[k] for k, v in losses.items()}
        return loss, losses

    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        out = self.model(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(to_floats(loss_dct), prog_bar=True, on_step=True)
        self.log("total", loss.float(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        out = self.model(batch)
        loss, losses = self.get_losses(out, batch)
        self.log("total", loss.float(), on_epoch=True)
        self.log_dict(to_floats(losses), on_epoch=True)
        # Cluster analysis: First collect graphs, then evaluate them later
        for mo_key, cf_key in ClusterFctType.required_model_outputs.items():
            self._cluster_scan_input[cf_key].append(out[mo_key].detach().cpu().numpy())
        if self.trainer.is_last_batch:
            self.evaluate_cluster_metrics()

    def evluate_cluster_metrics(self):
        metrics = {}
        cluster_result = self.cluster_scanner(
            **self._cluster_scan_input,
            epoch=self._epoch,
            start_params=self._best_cluster_params,
        )
        metrics |= cluster_result.metrics
        self._best_cluster_params = cluster_result.best_params
        # todo: Generalize for multiple cluster scanners
        metrics |= {
            f"best_dbscan_{param}": val
            for param, val in cluster_result.best_params.items()
        }
