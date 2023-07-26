from pathlib import Path
from typing import Callable

import yaml
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from rich.console import Console
from rich.table import Table

from gnn_tracking.utils.dictionaries import to_floats
from gnn_tracking.utils.log import logger


def format_results_table(
    results: dict[str, float],
    *,
    header: str = "",
    printed_results_filter: Callable[[str], bool] | None = None,
    highlight_metric: Callable[[str], bool] | None = None,
) -> Table:
    """Format a dictionary of results as a rich table.

    Args:
        results: Dictionary of results
        header: Header to prepend to the log message
        printed_results_filter: Function that takes a metric name and returns
            whether it should be printed in the log output.
            If None: Print everything
        highlight_metric: Function that takes a metric name and returns
            whether it should be highlighted in the log output.
            If None: Don't highlight anything

    Returns:
        Rich table
    """
    table = Table(title=header)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Error", justify="right")
    results = dict(sorted(results.items()))  # type: ignore
    for k, v in results.items():
        if printed_results_filter is not None and not printed_results_filter(k):
            continue
        if k.endswith("_std"):
            continue
        style = None
        if highlight_metric is not None and highlight_metric(k):
            style = "bright_magenta bold"
        err = results.get(f"{k}_std", float("nan"))
        table.add_row(k, f"{v:.5f}", f"{err:.5f}", style=style)
    return table


class PrintValidationMetrics(Callback):
    """This callback prints the validation metrics after every epoch.

    If the lightning module has a `printed_results_filter` attribute, only
    metrics for which this function returns True are printed.
    If the lightning module has a `highlight_metric` attribute, the metric
    returned by this function is highlighted in the output.
    """

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if not metrics:
            return
        console = Console()
        with console.capture() as capture:
            console.print(
                format_results_table(
                    to_floats(metrics),
                    header=f"Validation epoch={trainer.current_epoch}",
                    printed_results_filter=getattr(
                        pl_module, "printed_results_filter", None
                    ),
                    highlight_metric=getattr(pl_module, "highlight_metric", None),
                )
            )
        pl_module.print(capture.get())


class ExpandWandbConfig(Callback):
    """This callback adds more information to Weights & Biases hyperparameters:

    * Information about the optimizer/scheduler.
    * Information from the datamodule

    This will also avoid problems where hyperparameters are not synced at the beginning
    (but only at the end, in particular failing to save them if the run is interrupted).
    """

    def __init__(
        self,
    ):
        self._wandb_logger = None
        self._tensorboard_logger = None

    def _find_loggers(self, trainer: Trainer) -> None:
        for l in trainer.loggers:
            if isinstance(l, TensorBoardLogger):
                self._tensorboard_logger = l
            elif isinstance(l, WandbLogger):
                self._wandb_logger = l
        if not self._tensorboard_logger:
            logger.warning(
                "No TensorBoardLogger found. Will not be able to log full config to"
                " wandb"
            )
        if not self._wandb_logger:
            logger.warning("No WandbLogger found. Will not be able to log to wandb.")

    def _get_config(self) -> dict:
        if not self._tensorboard_logger:
            return {}
        config_path = Path(self._tensorboard_logger.log_dir, "config.yaml")
        if not config_path.exists():
            logger.warning("Could not find config.yaml in log directory")
        with config_path.open() as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._find_loggers(trainer)
        if self._wandb_logger is None:
            return
        config = self._get_config()
        assert self._wandb_logger
        self._wandb_logger.log_hyperparams(
            {
                "optimizer": config["model"].get("init_args", {}).get("optimizer"),
                "scheduler": config["model"].get("init_args", {}).get("scheduler"),
                "data": trainer.datamodule.hparams,
                **pl_module.hparams,
            }
        )
