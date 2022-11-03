from __future__ import annotations

import collections
import logging
import os
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import pandas as pd
import tabulate
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_tracking.metrics.binary_classification import BinaryClassificationStats
from gnn_tracking.postprocessing.clusterscanner import ClusterScanResult
from gnn_tracking.training.dynamiclossweights import (
    ConstantLossWeights,
    DynamicLossWeights,
)
from gnn_tracking.utils.device import guess_device
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.nomenclature import denote_pt
from gnn_tracking.utils.timing import timing

#: Function type that can be used as hook for the training/test step in the
#: `TCNTrainer` class. The function takes the trainer instance as first argument and
#: a dictionary of losses/metrics as second argument.
hook_type = Callable[["TCNTrainer", dict[str, Tensor]], None]


class LossFctType(Protocol):
    """Type of a loss function"""

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        ...

    def to(self, device: torch.device) -> LossFctType:
        ...


class ClusterFctType(Protocol):
    """Type of a clustering scanner function"""

    def __call__(
        self,
        graphs: list[np.ndarray],
        truth: list[np.ndarray],
        sectors: list[np.ndarray],
        pts: list[np.ndarray],
        epoch=None,
        start_params: dict[str, Any] | None = None,
    ) -> ClusterScanResult:
        ...


# The following abbreviations are used throughout the code:
# W: edge weights
# B: condensation likelihoods
# H: clustering coordinates
# Y: edge truth labels
# L: hit truth labels
# P: Track parameters
class TCNTrainer:
    def __init__(
        self,
        model,
        loaders: dict[str, DataLoader],
        loss_functions: dict[str, LossFctType],
        *,
        device=None,
        lr: Any = 5e-4,
        optimizer: Callable = Adam,
        lr_scheduler: None | Callable = None,
        loss_weights: dict[str, float] | DynamicLossWeights | None = None,
        cluster_functions: dict[str, ClusterFctType] | None = None,
    ):
        """Main trainer class of the condensation network approach.

        Args:
            model:
            loaders:
            loss_functions: Dictionary of loss functions, keyed by loss name
            device:
            lr: Learning rate
            optimizer: Optimizer to use (default: Adam): Function. Will be called with
                the model parameters as first positional parameter and with the learning
                rate as keyword argument (``lr``).
            lr_scheduler: Learning rate scheduler. If it needs parameters, apply
                ``functools.partial`` first
            loss_weights: Weight different loss functions.
                Either `DynamicLossWeights` object or a dictionary of weights keyed by
                loss name.
                If a dictionary and a key is left out, the weight is set to 1.0.
                The weights will be normalized to sum to 1.0 before use.
                If one of the loss functions called ``l`` returns a dictionary with keys
                k, the keys for loss_weights should be ``k_l``.
            cluster_functions: Dictionary of functions that take the output of the model
                during testing and report additional figures of merits (e.g.,
                clustering)
        """
        self.logger = get_logger("TCNTrainer", level=logging.INFO)
        self.device = guess_device(device)
        del device
        self.logger.info("Using device %s", self.device)
        #: Checkpoints are saved to this directory by default
        self.checkpoint_dir = Path(".")
        self.model = model.to(self.device)
        self.train_loader = loaders["train"]
        self.test_loader = loaders["test"]
        self.val_loader = loaders["val"]

        self.loss_functions = {k: v.to(self.device) for k, v in loss_functions.items()}
        if cluster_functions is None:
            cluster_functions = {}
        self.clustering_functions = cluster_functions

        if isinstance(loss_weights, DynamicLossWeights):
            self._loss_weight_setter = loss_weights
        elif isinstance(loss_weights, dict) or loss_weights is None:
            self._loss_weight_setter = ConstantLossWeights(loss_weights=loss_weights)
        else:
            raise ValueError("Invalid value for loss_weights.")

        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self._lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else None

        # Current epoch
        self._epoch = 0

        #: Hooks to be called after training epoch (please use `add_hook` to add them)
        self._train_hooks: list[hook_type] = []
        #: Hooks to be called after testing (please use `add_hook` to add them)
        self._test_hooks: list[hook_type] = []

        #: Mapping of cluster function name to best parameter
        self._best_cluster_params: dict[str, dict[str, Any] | None] = {}

        # output quantities
        self.train_loss: list[pd.DataFrame] = []
        self.test_loss: list[pd.DataFrame] = []

        #: Number of batches that are being used for the clustering functions and the
        #: evaluation of the related metrics.
        self.max_batches_for_clustering = 10

        #: pT thresholds that are being used in the evaluation of metrics in the test
        #: step
        self.pt_thlds = [0.9, 1.5]

    def add_hook(self, hook: hook_type, called_at: str) -> None:
        """Add a hook to training/test step

        Args:
            hook: Callable that takes a training model and a dictionary of tensors as
                inputs
            called_at: train or test

        Returns:
            None

        Example:


        """
        if called_at == "train":
            self._train_hooks.append(hook)
        elif called_at == "test":
            self._test_hooks.append(hook)
        else:
            raise ValueError("Invalid value for called_at")

    def evaluate_model(self, data: Data, mask_pids_reco=True) -> dict[str, Tensor]:
        """Evaluate the model on the data and return a dictionary of outputs

        Args:
            data:
            mask_pids_reco: If True, mask out PIDs for non-reconstructables
        """
        data = data.to(self.device)
        out = self.model(data)
        if mask_pids_reco:
            pid_field = data.particle_id * data.reconstructable.long()
        else:
            pid_field = data.particle_id
        dct = {
            "w": out["W"].squeeze() if out["W"] is not None else None,
            "x": out["H"],
            "beta": out["B"].squeeze(),
            "y": data.y,
            "particle_id": pid_field,
            "track_params": data.pt,
            "pt": data.pt,
            "reconstructable": data.reconstructable.long(),
            "pred": out["P"],
        }
        return dct

    def get_batch_losses(
        self, model_output: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Calculate the losses for a batch of data

        Args:
            model_output:

        Returns:
            total loss, dictionary of losses, where total loss includes the weights
            assigned to the individual losses
        """
        individual_losses = {}
        for key, loss_func in self.loss_functions.items():
            loss = loss_func(**model_output)
            if isinstance(loss, Mapping):
                for k, v in loss.items():
                    individual_losses[f"{key}_{k}"] = v
            else:
                individual_losses[key] = loss

        total = sum(
            self._loss_weight_setter[k] * individual_losses[k]
            for k in individual_losses.keys()
        )
        if torch.isnan(total):
            raise RuntimeError("NaN loss encountered in test step")
        return total, individual_losses

    def _log_losses(
        self,
        batch_losses: dict[str, Tensor | float],
        *,
        style="table",
        header: str = "",
    ) -> None:
        """Log the losses

        Args:
            batch_loss: Total loss
            batch_losses:
            style: "table" or "inline"
            header: Header to prepend to the log message

        Returns:
            None
        """
        if header:
            report_str = header
        else:
            report_str = ""
        if style == "table":
            report_str += "\n"
        table_items: list[tuple[str, float]] = sorted(batch_losses.items())
        if style == "table":
            report_str += tabulate.tabulate(
                table_items,
                tablefmt="outline",
                floatfmt=".5f",
                headers=["Metric", "Value"],
            )
        else:
            report_str += ", ".join(f"{k}={v:>10.5f}" for k, v in table_items)
        self.logger.info(report_str)

    def train_step(self, *, max_batches: int | None = None) -> dict[str, float]:
        """

        Args:
            max_batches:  Only process this many batches per epoch (useful for testing
                to get to the validation step more quickly)

        Returns:

        """
        self.model.train()
        _losses = collections.defaultdict(list)
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            if max_batches and batch_idx > max_batches:
                break
            model_output = self.evaluate_model(data)
            batch_loss, batch_losses = self.get_batch_losses(model_output)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            if not (batch_idx % 10):
                self._log_losses(
                    batch_losses,
                    header=f"Epoch {self._epoch:>2} "
                    f"({batch_idx:>5}/{len(self.train_loader)}): ",
                    style="inline",
                )

            _losses["total"].append(batch_loss.item())
            for key, loss in batch_losses.items():
                _losses[f"{key}"].append(loss.item())
                _losses[f"{key}_weighted"].append(
                    loss.item() * self._loss_weight_setter[key]
                )

        losses = {k: np.nanmean(v) for k, v in _losses.items()}
        self._loss_weight_setter.step(losses)
        self.train_loss.append(pd.DataFrame(losses, index=[self._epoch]))
        for hook in self._train_hooks:
            hook(self, losses)
        return losses

    def _edge_pt_mask(self, edge_index: Tensor, pt: Tensor, pt_min=0.0) -> Tensor:
        pt_a = pt[edge_index[0]]
        pt_b = pt[edge_index[1]]
        return (pt_a > pt_min) | (pt_b > pt_min)

    def test_step(self, thld=0.5, val=True) -> dict[str, float]:
        """Test the model on the validation or test set

        Args:
            thld: Threshold for edge classification
            val: Use validation dataset rather than test dataset

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        # Objects in the following three lists are used for clustering
        graphs: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        sectors: list[np.ndarray] = []
        pts: list[np.ndarray] = []

        batch_losses = collections.defaultdict(list)
        with torch.no_grad():
            loader = self.val_loader if val else self.test_loader
            for _batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                model_output = self.evaluate_model(data, mask_pids_reco=False)
                batch_loss, these_batch_losses = self.get_batch_losses(model_output)

                if model_output["w"] is not None:
                    for pt_min in self.pt_thlds:
                        edge_pt_mask = self._edge_pt_mask(
                            data.edge_index, data.pt, pt_min
                        )
                        bcs = BinaryClassificationStats(
                            output=model_output["w"][edge_pt_mask],
                            y=model_output["y"][edge_pt_mask].long(),
                            thld=thld,
                        )
                        for k, v in bcs.get_all().items():
                            batch_losses[denote_pt(k, pt_min)].append(v)

                batch_losses["total"].append(batch_loss.item())
                for key, loss in these_batch_losses.items():
                    batch_losses[key].append(loss.item())
                    batch_losses[f"{key}_weighted"].append(
                        loss.item() * self._loss_weight_setter[key]
                    )

                if (
                    self.clustering_functions
                    and _batch_idx <= self.max_batches_for_clustering
                ):
                    graphs.append(model_output["x"].detach().cpu().numpy())
                    truths.append(model_output["particle_id"].detach().cpu().numpy())
                    sectors.append(data.sector.detach().cpu().numpy())
                    pts.append(model_output["pt"].detach().cpu().numpy())

        losses = {k: np.nanmean(v) for k, v in batch_losses.items()}
        for k, f in self.clustering_functions.items():
            cluster_result = f(
                graphs,
                truths,
                sectors,
                pts,
                epoch=self._epoch,
                start_params=self._best_cluster_params.get(k, None),
            )
            if cluster_result is not None:
                losses.update(cluster_result.metrics)
                self._best_cluster_params[k] = cluster_result.best_params
                losses.update(
                    {
                        f"best_{k}_{param}": val
                        for param, val in cluster_result.best_params.items()
                    }
                )

        self.test_loss.append(pd.DataFrame(losses, index=[self._epoch]))
        for hook in self._test_hooks:
            hook(self, losses)
        return losses

    def step(self, *, max_batches: int | None = None) -> dict[str, float]:
        """Train one epoch and test

        Args:
            max_batches: See train_step
        """
        self._epoch += 1
        with timing(f"Training for epoch {self._epoch}"):
            train_losses = self.train_step(max_batches=max_batches)
        with timing(f"Test step for epoch {self._epoch}"):
            test_results = self.test_step(thld=0.5, val=True)
        results = {
            **{f"{k}_train": v for k, v in train_losses.items()},
            **test_results,
        }
        self._log_losses(
            results,
            header=f"Results {self._epoch}: ",
        )
        if self._lr_scheduler:
            self._lr_scheduler.step()
        return results

    def train(self, epochs=1000, max_batches: int | None = None):
        """Train the model.

        Args:
            epochs:
            max_batches: See train_step.

        Returns:

        """
        for _ in range(1, epochs + 1):
            try:
                self.step(max_batches=max_batches)
            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt")
                self.save_checkpoint()
                raise
        self.save_checkpoint()

    # noinspection PyMethodMayBeStatic
    def get_checkpoint_name(self) -> str:
        """Generate name of checkpoint file based on current time."""
        now = datetime.now()
        return f"{now:%y%m%d_%H%M%S}_model.pt"

    def get_checkpoint_path(self, path: str | PurePath = "") -> Path:
        """Get checkpoint path based on user input."""
        if not path:
            return self.checkpoint_dir / self.get_checkpoint_name()
        if isinstance(path, str) and os.sep not in path:
            return self.checkpoint_dir / path
        return Path(path)

    def save_checkpoint(self, path: str | PurePath = "") -> None:
        """Save state of model, optimizer and more for later resuming of training."""
        path = self.get_checkpoint_path(path)
        self.logger.info(f"Saving checkpoint to {path}")
        torch.save(
            {
                "epoch": self._epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | PurePath, device=None) -> None:
        """Resume training from checkpoint"""
        checkpoint = torch.load(self.get_checkpoint_path(path), map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"]
