"""Base class used for all pytorch lightning modules."""
import collections
import logging
from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from rich.console import Console
from rich.table import Table
from torch import Tensor, nn
from torch_geometric.data import Data

from gnn_tracking.utils.dictionaries import to_floats
from gnn_tracking.utils.lightning import StandardError, obj_from_or_to_hparams
from gnn_tracking.utils.log import get_logger

# The following abbreviations are used throughout the code:
# W: edge weights
# B: condensation likelihoods
# H: clustering coordinates
# Y: edge truth labels
# L: hit truth labels
# P: Track parameters


class ImprovedLogLM(LightningModule):
    def __init__(self, **kwargs):
        """This subclass of `LightningModule` adds some convenience to logging,
        e.g., logging of statistical uncertainties (batch-to-batch) and logging
        of the validation metrics to the console after each validation epoch.
        """
        super().__init__(**kwargs)
        self._uncertainties = collections.defaultdict(StandardError)

    def log_dict_with_errors(self, dct: dict[str, float]) -> None:
        """Log a dictionary of values with their statistical uncertainties.

        This method only starts calculating the uncertainties. To log them,
        `_log_errors` needs to be called at the end of the train/val/test epoch
        (done with the hooks configured in this class).
        """
        self.log_dict(
            dct,
            on_epoch=True,
        )
        for k, v in dct.items():
            if f"{k}_std" in dct:
                continue
            self._uncertainties[k](torch.Tensor([v]))

    def _log_errors(self) -> None:
        """Log the uncertainties calculated in `log_dict_with_errors`.
        Needs to be called at the end of the train/val/test epoch.
        """
        for k, v in self._uncertainties.items():
            self.log(k + "_std", v.compute(), on_epoch=True)

    # noinspection PyUnusedLocal
    def on_training_epoch_end(self, *args) -> None:
        self._log_errors()

    def on_validation_epoch_end(self) -> None:
        self._log_errors()

    def on_test_epoch_end(self) -> None:
        self._log_errors()

    def format_results_table(
        self,
        results: dict[str, float],
        *,
        header: str = "",
    ) -> Table:
        """Format a dictionary of results as a rich table.

        Args:
            results: Dictionary of results
            header: Header to prepend to the log message

        Returns:
            None
        """
        table = Table(title=header)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Error", justify="right")
        results = dict(sorted(results.items()))  # type: ignore
        for k, v in results.items():
            if not self.printed_results_filter(k):
                continue
            if k.endswith("_std"):
                continue
            style = None
            if self.highlight_metric(k):
                style = "bright_magenta bold"
            err = results.get(f"{k}_std", float("nan"))
            table.add_row(k, f"{v:.5f}", f"{err:.5f}", style=style)
        return table

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def printed_results_filter(self, key: str) -> bool:
        """Should a metric be printed in the log output for the val/test step?

        This is meant to be overridden by your personal trainer.
        """
        return True

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def highlight_metric(self, metric: str) -> bool:
        """Should a metric be highlighted in the log output for the val/test step?"""
        return False

    def on_validation_end(self, *args, **kwargs) -> None:
        metrics = self.trainer.callback_metrics
        if not metrics:
            return
        console = Console()
        console.print("\n")
        console.print(
            self.format_results_table(
                to_floats(metrics), header=f"Validation epoch={self.current_epoch}"
            )
        )


class TrackingModule(ImprovedLogLM):
    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        preproc: nn.Module | None = None,
    ):
        """Base class for all pytorch lightning modules in this project."""
        super().__init__()
        self.model = obj_from_or_to_hparams(self, "model", model)
        self.logg = get_logger("TM", level=logging.DEBUG)
        self.preproc = obj_from_or_to_hparams(self, "preproc", preproc)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, data: Data) -> Tensor:
        return self.model.forward(data)

    def data_preproc(self, data: Data) -> Data:
        print("preproc", data)
        if self.preproc is not None:
            return self.preproc(data)
        return data

    def test_step(self, batch: Data, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
