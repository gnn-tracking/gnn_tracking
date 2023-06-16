import logging
from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from rich import Console
from rich.table import Table
from torch import Tensor

from gnn_tracking.utils.log import get_logger

# class SuppressOOMExceptions:
#     def __init__(self, trainer):
#         self._trainer = trainer
#
#     def __enter__(self):
#         ...
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         if exc_type == RuntimeError and "out of memory" in str(exc_value):
#             self._trainer.logg.warning(
#                 "WARNING: ran out of memory (OOM), skipping batch. "
#                 "If this happens frequently, decrease the batch size. "
#                 "Will abort if we get 10 consecutive OOM errors."
#             )
#             self._trainer._n_oom_errors_in_a_row += 1
#             return self._trainer._n_oom_errors_in_a_row < 10
#         return False

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
