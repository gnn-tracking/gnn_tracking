from __future__ import annotations

import collections
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from gnn_tracking.utils.losses import (
    BackgroundLoss,
    EdgeWeightLoss,
    ObjectLoss,
    PotentialLoss,
)
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
        device="cpu",
        lr: Any = 5 * 10**-4,
        q_min=0.01,
        sb=1,
        object_loss_mode="purity",
        predict_track_params=False,
        lr_scheduler: None | Callable = None,
    ):
        """

        Args:
            model:
            loaders:
            device:
            lr: Learning rate
            q_min:
            sb:
            object_loss_mode:
            predict_track_params:
            lr_scheduler: Learning rate scheduler. If it needs parameters, apply
                functools.partial first
        """
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.test_loader = loaders["test"]
        self.val_loader = loaders["val"]
        self.device = device
        self.edge_weight_loss = EdgeWeightLoss().to(device)
        self.potential_loss = PotentialLoss(q_min=q_min, device=device)
        self.background_loss = BackgroundLoss(device=device, sb=sb)
        self.object_loss = ObjectLoss(device=device, mode=object_loss_mode)
        self.predict_track_params = predict_track_params

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self._lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else None

        # output quantities
        self.train_loss = []
        self.test_loss = []

    def train_step(self, epoch: int):
        self.model.train()

        losses = collections.defaultdict(list)
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            if self.predict_track_params:
                W, H, B, P = self.model(data.x, data.edge_index, data.edge_attr)
            else:
                W, H, B = self.model(data.x, data.edge_index, data.edge_attr)
            Y, W = data.y, W.squeeze(1)
            L, T = data.particle_id, data.pt
            R = data.reconstructable.long()
            L = L * R  # should we mask out non-reconstructables?
            B = B.squeeze()
            loss_W = self.edge_weight_loss(W, B, H, Y, L)
            loss_V = self.potential_loss(W, B, H, Y, L)
            loss_B = self.background_loss(W, B, H, Y, L)

            loss = loss_W + loss_V + loss_B
            if self.predict_track_params:
                loss_P = self.object_loss(W, B, H, P, Y, L, T, R)
                loss += loss_P

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if not (batch_idx % 10):
                print(
                    f"Epoch {epoch} ({batch_idx}/{len(self.train_loader)}):"
                    + f" loss={loss.item():.5f}; loss_W={loss_W.item():.5f};"
                    + f" loss_V={loss_V.item():.5f}; loss_B={loss_B.item():.5f};"
                )

            losses["total"].append(loss.item())
            losses["W"].append(loss_W.item())
            losses["V"].append(loss_V.item())
            losses["B"].append(loss_B.item())
            # losses['P'].append(loss_P.item())

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        self.train_loss.append(pd.DataFrame(losses, index=[epoch]))

    def test_step(self, epoch: int, thld=0.5):
        self.model.eval()
        losses = collections.defaultdict(list)
        with torch.no_grad():
            for _batch_idx, data in enumerate(self.test_loader):
                data = data.to(self.device)
                if self.predict_track_params:
                    W, H, B, P = self.model(data.x, data.edge_index, data.edge_attr)
                else:
                    W, H, B = self.model(data.x, data.edge_index, data.edge_attr)
                Y, W = data.y, W.squeeze(1)
                L, T = data.particle_id, data.pt
                R = data.reconstructable.long()
                B = B.squeeze()
                loss_W = self.edge_weight_loss(W, B, H, Y, L).item()
                loss_V = self.potential_loss(W, B, H, Y, L).item()
                loss_B = self.background_loss(W, B, H, Y, L).item()
                if self.predict_track_params:
                    loss_P = self.object_loss(W, B, H, P, Y, L, T, R).item()
                    losses["P"].append(loss_P)
                bcs = BinaryClassificationStats(W, Y.long(), thld)
                losses["total"].append(loss_W + loss_V + loss_B)
                losses["W"].append(loss_W)
                losses["V"].append(loss_V)
                losses["B"].append(loss_B)
                losses["acc"].append(bcs.acc)

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        print("test", losses)
        self.test_loss.append(pd.DataFrame(losses, index=[epoch]))

    def validate(self) -> float:
        """

        Returns:
            Optimal threshold for binary classification.
        """
        self.model.eval()
        # Optimal threshold for binary classification per batch
        opt_thlds = []
        for data in iter(self.val_loader):
            data = data.to(self.device)
            if self.predict_track_params:
                W, H, B, P = self.model(data.x, data.edge_index, data.edge_attr)
            else:
                W, H, B = self.model(data.x, data.edge_index, data.edge_attr)
            Y, W = data.y, W.squeeze(1)
            diff, opt_thld = 100, 0
            for thld in np.arange(0.01, 0.5, 0.01):
                bcs = BinaryClassificationStats(W, Y.long(), thld)
                delta = abs(bcs.TPR - bcs.TNR)
                if delta < diff:
                    diff, opt_thld = delta, thld
            opt_thlds.append(opt_thld)
        return np.nanmean(opt_thlds).item()

    def train(self, epochs=1000):
        for epoch in range(1, epochs + 1):
            print(f"---- Epoch {epoch} ----")
            self.train_step(epoch)
            thld = self.validate()
            self.test_step(epoch, thld=thld)
            if self._lr_scheduler:
                self._lr_scheduler.step()
