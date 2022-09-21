from __future__ import annotations

import collections
import logging
from typing import Any, Callable, DefaultDict, Mapping, Protocol

import numpy as np
import pandas as pd
import tabulate
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_tracking.postprocessing.clusterscanner import ClusterScanResult
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.training import BinaryClassificationStats

hook_type = Callable[[torch.nn.Module, dict[str, Tensor]], None]
loss_fct_type = Callable[..., Tensor]


class ClusterFctType(Protocol):
    def __call__(
        self,
        graphs: list[np.ndarray],
        truth: list[np.ndarray],
        sectors: list[np.ndarray],
        epoch=None,
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
        loss_functions: dict[str, loss_fct_type],
        *,
        device="cpu",
        lr: Any = 5 * 10**-4,
        lr_scheduler: None | Callable = None,
        loss_weights: dict[str, float] = None,
        cluster_functions: dict[str, ClusterFctType] | None = None,
    ):
        """

        Args:
            model:
            loaders:
            loss_functions: Dictionary of loss functions, keyed by loss name
            device:
            lr: Learning rate
            lr_scheduler: Learning rate scheduler. If it needs parameters, apply
                functools.partial first
            loss_weights: Weight different loss functions. If a key is left out, the
                weight is set to 1.0. The weights will be normalized to sum to 1.0
                before use.
                If one of the loss functions called ``l`` returns a dictionary with keys
                k, the keys for loss_weights should be ``k_l``.
            cluster_functions: Dictionary of functions that take the output of the model
                during testing and report additional figures of merits (e.g.,
                clustering)
        """
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.test_loader = loaders["test"]
        self.val_loader = loaders["val"]
        self.device = device

        self.loss_functions = loss_functions
        if cluster_functions is None:
            cluster_functions = {}
        self.clustering_functions = cluster_functions

        self._loss_weights: DefaultDict[str, float] = collections.defaultdict(
            lambda: 1.0
        )
        if loss_weights is not None:
            self._loss_weights.update(loss_weights)

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self._lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else None

        # Current epoch
        self._epoch = 0

        self._train_hooks: list[hook_type] = []
        self._test_hooks: list[hook_type] = []

        self.logger = get_logger("TCNTrainer", level=logging.INFO)

        # output quantities
        self.train_loss: list[pd.DataFrame] = []
        self.test_loss: list[pd.DataFrame] = []

        self.max_batches_for_clustering = 10

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

        assert set(self._loss_weights).issubset(set(individual_losses))

        # Note that we take the keys from individual_losses and not from
        # self._loss_weights (because that is a defaultdict and might not have all keys,
        # yet).
        # total_weight = sum(self._loss_weights[k] for k in individual_losses)

        total = sum(
            # self._loss_weights[k] / total_weight * individual_losses[k]
            self._loss_weights[k] * individual_losses[k]
            for k in individual_losses.keys()
        )
        return total, individual_losses

    def _log_losses(
        self,
        batch_loss: Tensor,
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
        table_items: list[tuple[str, float]] = [("Total", batch_loss.item())]
        for k, v in batch_losses.items():
            if k.casefold() == "total":
                continue
            if k in self._loss_weights:
                table_items.append((k, float(v) * self._loss_weights[k]))
            else:
                table_items.append((k, float(v)))
        if style == "table":
            report_str += tabulate.tabulate(
                table_items, tablefmt="fancy_grid", floatfmt=".5f"
            )
        else:
            report_str += ", ".join(f"{k}={v:>9.5f}" for k, v in table_items)
        self.logger.info(report_str)

    def train_step(self, *, max_batches: int | None = None):
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
                    batch_loss,
                    batch_losses,
                    header=f"Epoch {self._epoch} "
                    f"({batch_idx:>5}/{len(self.train_loader)}): ",
                    style="inline",
                )

            _losses["total"].append(batch_loss.item())
            for key, loss in batch_losses.items():
                _losses[key].append(loss.item())

        losses = {k: np.nanmean(v) for k, v in _losses.items()}
        self.train_loss.append(pd.DataFrame(losses, index=[self._epoch]))
        for hook in self._train_hooks:
            hook(self.model, losses)

    def test_step(self, thld=0.5, val=True) -> dict[str, float]:
        self.model.eval()
        losses = collections.defaultdict(list)

        graphs: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        sectors: list[np.ndarray] = []
        with torch.no_grad():
            loader = self.val_loader if val else self.test_loader
            for _batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                model_output = self.evaluate_model(data, mask_pids_reco=False)
                batch_loss, batch_losses = self.get_batch_losses(model_output)

                if model_output["w"] is not None:
                    bcs = BinaryClassificationStats(
                        output=model_output["w"], y=model_output["y"].long(), thld=thld
                    )
                    losses["acc"].append(bcs.acc)
                    losses["TPR"].append(bcs.TPR)
                    losses["TNR"].append(bcs.TNR)
                    losses["FPR"].append(bcs.FPR)
                    losses["FNR"].append(bcs.FNR)

                losses["total"].append(batch_loss.item())
                for key, loss in batch_losses.items():
                    losses[key].append(loss.item())

                if _batch_idx <= self.max_batches_for_clustering:
                    graphs.append(model_output["x"].detach().cpu().numpy())
                    truths.append(model_output["particle_id"].detach().cpu().numpy())
                    sectors.append(data.sector.detach().cpu().numpy())

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        for f in self.clustering_functions.values():
            cluster_result = f(graphs, truths, sectors, epoch=self._epoch)
            if cluster_result is not None:
                losses.update(cluster_result.metrics)
        self._log_losses(losses["total"], losses, header=f"Test {self._epoch}: ")
        self.test_loss.append(pd.DataFrame(losses, index=[self._epoch]))
        for hook in self._test_hooks:
            hook(self.model, losses)
        return losses

    def train(self, epochs=1000, max_batches: int | None = None):
        """

        Args:
            epochs:
            max_batches: See train_step.

        Returns:

        """
        for _ in range(1, epochs + 1):
            self._epoch += 1
            self.logger.info(f"---- Epoch {self._epoch} ----")
            self.train_step(max_batches=max_batches)
            self.test_step(thld=0.5, val=True)
            if self._lr_scheduler:
                self._lr_scheduler.step()
