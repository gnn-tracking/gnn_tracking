import collections
import logging
from typing import Any, Iterable

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.cli import LightningCLI, LRSchedulerCallable, OptimizerCallable
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch import Tensor as T
from torch import nn
from torch_geometric.data import Data

from gnn_tracking.metrics.binary_classification import (
    BinaryClassificationStats,
    get_maximized_bcs,
    roc_auc_score,
)
from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    GraphConstructionHingeEmbeddingLoss,
    PotentialLoss,
)
from gnn_tracking.postprocessing.clusterscanner import ClusterFctType
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.nomenclature import denote_pt

# Also add this to EC metrics
# | get_maximized_bcs(output=predicted, y=true)


def get_roc_auc_scores(true, predicted, fprs: Iterable[float | None]):
    metrics = {}
    metrics["roc_auc"] = roc_auc_score(y_true=true, y_score=predicted)
    for max_fpr in fprs:
        metrics[f"roc_auc_{max_fpr}FPR"] = roc_auc_score(
            y_true=true,
            y_score=predicted,
            max_fpr=max_fpr,
        )
    return metrics


def get_binary_classification_metrics_at_thld(
    *, edge_index: T, pt: T, w: T, y: T, pt_min: float, thld: float
) -> dict[str, float]:
    """Evaluate edge classification metrics for a given pt threshold and
    EC threshold.

    Args:
        pt_min: pt threshold: We discard all edges where both nodes have
            `pt <= pt_min` before evaluating any metric.
        thld: EC threshold

    Returns:
        Dictionary of metrics
    """
    pt_a = pt[edge_index[0]]
    pt_b = pt[edge_index[1]]
    edge_pt_mask = (pt_a > pt_min) | (pt_b > pt_min)

    predicted = w[edge_pt_mask]
    true = y[edge_pt_mask].long()

    bcs = BinaryClassificationStats(
        output=predicted,
        y=true,
        thld=thld,
    )
    metrics = bcs.get_all()
    return {denote_pt(k, pt_min): v for k, v in metrics.items()}


class SuppressOOMExceptions:
    def __init__(self, trainer):
        self._trainer = trainer

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == RuntimeError and "out of memory" in str(exc_value):
            self._trainer.logg.warning(
                "WARNING: ran out of memory (OOM), skipping batch. "
                "If this happens frequently, decrease the batch size. "
                "Will abort if we get 10 consecutive OOM errors."
            )
            self._trainer._n_oom_errors_in_a_row += 1
            return self._trainer._n_oom_errors_in_a_row < 10
        return False


# The following abbreviations are used throughout the code:
# W: edge weights
# B: condensation likelihoods
# H: clustering coordinates
# Y: edge truth labels
# L: hit truth labels
# P: Track parameters


class TrackingModule(LightningModule):
    def __init__(
        self,
        model: LightningModule,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ):
        super().__init__()
        self.logg = get_logger("TM", level=logging.DEBUG)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def test_step(self, batch, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    # --- All things logging ---

    def format_results_table(
        self,
        results: dict[str, Tensor | float],
        *,
        header: str = "",
    ) -> Table:
        """Log the losses

        Args:
            results:
            header: Header to prepend to the log message

        Returns:
            None
        """
        table = Table(title=header)
        table.add_column("Metric")
        table.add_column("Value")
        table.add_column("Error")

        for k, v in results.items():
            if not self.printed_results_filter(k):
                continue
            if k.endswith("_std"):
                continue
            style = None
            if self.highlight_metric(k):
                style = "bold"
            err = results.get(f"{k}_std", float("nan"))
            table.add_row(k, f"{v:.5f}", f"{err:.5f}", style=style)

        return table

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def printed_results_filter(self, key: str) -> bool:
        """Should a metric be printed in the log output?

        This is meant to be overridden by your personal trainer.
        """
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def highlight_metric(self, metric: str) -> bool:
        """Should a metric be highlighted in the log output?"""
        return False

    def on_validation_end(self, *args, **kwargs) -> None:
        # Don't use on_validation_epoch_end, you'll be off by a linebreak
        metrics = self.trainer.callback_metrics
        if not metrics:
            return
        console = Console()
        console.print("\n")
        console.print(self.format_results_table(metrics, header="Validation"))


def to_floats(inpt):
    if isinstance(inpt, dict):
        return {k: to_floats(v) for k, v in inpt.items()}
    elif isinstance(inpt, list):
        return [to_floats(v) for v in inpt]
    elif isinstance(inpt, torch.Tensor):
        return inpt.float()
    return inpt


class MLModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        loss_fct: GraphConstructionHingeEmbeddingLoss,
        lw_repulsive: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters("lw_repulsive")
        self.loss_fct = loss_fct

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
        out = self.model(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(loss_dct, prog_bar=True, on_step=True)
        self.log("total", loss.float(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, bach_idx: int):
        out = self.model(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(to_floats(loss_dct), on_epoch=True)
        self.log("total", loss.float(), on_epoch=True)
        # todo: add graph analysis

    def test_step(self, batch, batch_idx: int):
        self.validation_step(batch, batch_idx)


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
        metrics |= get_roc_auc_scores(
            true=batch.y, predicted=out["w"], fprs=[None, 0.01, 0.001]
        )
        metrics |= get_maximized_bcs(y=batch.y, output=out["w"])
        # todo: add graph analysis
        self.loc_dict(metrics, on_epoch=True)


class OCModule(TrackingModule):
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
        self.potential_loss = potential_loss
        self.background_loss = background_loss
        self.cluster_scanner = cluster_scanner
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


def cli_main():
    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa F841
        MLModule,
        datamodule_class=TrackingDataModule,
        trainer_defaults=dict(callbacks=[RichProgressBar(leave=True)]),
    )


if __name__ == "__main__":
    cli_main()
