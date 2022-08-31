from __future__ import annotations

import collections
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_tracking.utils.training import BinaryClassificationStats


# The following abbreviations are used throughout the code:
# W: edge weights
# B: condensation likelihoods
# H: clustering coordinates
# Y: edge truth labels
# L: hit truth labels
# P: Track parameters
class GraphTCNTrainer:
    def __init__(
        self,
        model,
        loaders: dict[str, DataLoader],
        loss_functions: dict[str, Callable[[Any], Tensor]],
        *,
        device="cpu",
        lr: Any = 5 * 10**-4,
        predict_track_params=False,
        lr_scheduler: None | Callable = None,
    ):
        """

        Args:
            model:
            loaders:
            loss_functions: Dictionary of loss functions, keyed by loss name
            device:
            lr: Learning rate
            predict_track_params:
            lr_scheduler: Learning rate scheduler. If it needs parameters, apply
                functools.partial first
        """
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.test_loader = loaders["test"]
        self.val_loader = loaders["val"]
        self.device = device

        self.predict_track_params = predict_track_params

        self.loss_functions = loss_functions

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self._lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else None

        # Current epoch
        self._epoch = 0

        # output quantities
        self.train_loss = []
        self.test_loss = []

    def evaluate_model(self, data: Data, mask_pids_reco=True) -> dict[str, Tensor]:
        """Evaluate the model on the data and return a dictionary of outputs

        Args:
            data:
            mask_pids_reco: If True, mask out PIDs for non-reconstructables
        """
        data = data.to(self.device)
        if self.predict_track_params:
            W, H, B, P = self.model(data.x, data.edge_index, data.edge_attr)
        else:
            W, H, B = self.model(data.x, data.edge_index, data.edge_attr)
        if mask_pids_reco:
            pid_field = data.particle_id * data.reconstructable.long()
        else:
            pid_field = data.particle_id
        dct = {
            "w": W.squeeze(),
            "x": H,
            "beta": B.squeeze(),
            "y": data.y,
            "particle_id": pid_field,
            "track_params": data.pt,
            "reconstructable": data.reconstructable.long(),
        }
        if self.predict_track_params:
            dct["pred"] = P
        return dct

    def get_batch_losses(
        self, model_output: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Calculate the losses for a batch of data

        Args:
            model_output:

        Returns:
            total loss, dictionary of losses
        """
        individual = {
            key: loss_func(**model_output)
            for key, loss_func in self.loss_functions.items()
        }
        total = sum(individual.values())
        return total, individual

    def train_step(self, *, max_batches: int | None = None):
        """

        Args:
            max_batches:  Only process this many batches per epoch (useful for testing
                to get to the validation step more quickly)

        Returns:

        """
        self.model.train()

        losses = collections.defaultdict(list)
        for batch_idx, data in enumerate(self.train_loader):
            if max_batches and batch_idx > max_batches:
                break
            model_output = self.evaluate_model(data)
            batch_loss, batch_losses = self.get_batch_losses(model_output)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            if not (batch_idx % 10):
                report_str = (
                    f"Epoch {self._epoch} ({batch_idx}/{len(self.train_loader)}): "
                )
                report_str += f"total: {batch_loss.item():.5f} "
                for key, loss in batch_losses.items():
                    report_str += f"{key}: {loss.item():.5f} "
                print(report_str)

            losses["total"].append(batch_loss.item())
            for key, loss in batch_losses.items():
                losses[key].append(loss.item())

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        self.train_loss.append(pd.DataFrame(losses, index=[self._epoch]))

    def test_step(self, thld=0.5):
        self.model.eval()
        losses = collections.defaultdict(list)
        with torch.no_grad():
            for _batch_idx, data in enumerate(self.test_loader):
                model_output = self.evaluate_model(data, mask_pids_reco=False)
                batch_loss, batch_losses = self.get_batch_losses(model_output)
                bcs = BinaryClassificationStats(
                    output=model_output["w"], y=model_output["y"].long(), thld=thld
                )
                losses["total"].append(batch_loss.item())
                for key, loss in batch_losses.items():
                    losses[key].append(loss.item())
                losses["acc"].append(bcs.acc)

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        print("test", losses)
        self.test_loss.append(pd.DataFrame(losses, index=[self._epoch]))

    def validate(self) -> float:
        """

        Returns:
            Optimal threshold for binary classification.
        """
        self.model.eval()
        # Optimal threshold for binary classification per batch
        opt_thlds = []
        for data in iter(self.val_loader):
            model_output = self.evaluate_model(data)
            diff, opt_thld = 100, 0
            for thld in np.arange(0.01, 0.5, 0.01):
                bcs = BinaryClassificationStats(
                    output=model_output["w"], y=model_output["y"].long(), thld=thld
                )
                delta = abs(bcs.TPR - bcs.TNR)
                if delta < diff:
                    diff, opt_thld = delta, thld
            opt_thlds.append(opt_thld)
        return np.nanmean(opt_thlds).item()

    def train(self, epochs=1000, max_batches: int | None = None):
        """

        Args:
            epochs:
            max_batches: See train_step.

        Returns:

        """
        for _ in range(1, epochs + 1):
            self._epoch += 1
            print(f"---- Epoch {self._epoch} ----")
            self.train_step(max_batches=max_batches)
            thld = self.validate()
            self.test_step(thld=thld)
            if self._lr_scheduler:
                self._lr_scheduler.step()
