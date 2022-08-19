from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from gnn_tracking.utils.losses import (
    BackgroundLoss,
    EdgeWeightLoss,
    ObjectLoss,
    PotentialLoss,
)
from gnn_tracking.utils.training import binary_classification_stats


class GraphTCNTrainer:
    def __init__(
        self,
        model,
        loaders,
        device="cpu",
        lr=5 * 10**-4,
        q_min=0.01,
        sb=1,
        epochs=1000,
        object_loss_mode="purity",
        predict_track_params=False,
    ):
        self.model = model.to(device)
        self.epochs = epochs
        self.train_loader = loaders["train"]
        self.test_loader = loaders["test"]
        self.val_loader = loaders["val"]
        self.device = device
        self.edge_weight_loss = EdgeWeightLoss().to(device)
        self.potential_loss = PotentialLoss(q_min=q_min, device=device)
        self.background_loss = BackgroundLoss(q_min=q_min, device=device, sb=sb)
        self.object_loss = ObjectLoss(
            q_min=q_min, device=device, sb=sb, mode=object_loss_mode
        )
        self.predict_track_params = predict_track_params

        # quantities to predict
        self.W = torch.empty(1, dtype=torch.float, device=device)  # edge weights
        self.B = torch.empty(
            1, dtype=torch.float, device=device  # condensation likelihoods
        )
        self.H = torch.empty(
            1, dtype=torch.float, device=device  # clustering coordinates
        )
        self.Y = torch.empty(1, dtype=torch.float, device=device)  # edge truth labels
        self.L = torch.empty(1, dtype=torch.float, device=device)  # hit truth labels
        self.P = torch.empty(1, dtype=torch.float, device=device)  # track parameters

        # build a constrained optimizer
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # output quantities
        self.train_loss = []
        self.test_loss = []

    def train_step(self, epoch):
        self.model.train()

        losses = {"W": [], "V": [], "B": [], "P": [], "total": []}
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            if self.predict_track_params:
                self.W, self.H, self.B, self.P = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
                self.T = data.pt
            else:
                self.W, self.H, self.B = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
            self.Y, self.W = data.y, self.W.squeeze(1)
            self.L, self.T = data.particle_id, data.pt
            self.R = data.reconstructable.long()
            self.L = self.L * self.R  # should we mask out non-reconstructables?
            self.B = self.B.squeeze()
            loss_W = self.edge_weight_loss(self.W, self.B, self.H, self.Y, self.L)
            loss_V = self.potential_loss(self.W, self.B, self.H, self.Y, self.L)
            loss_B = self.background_loss(self.W, self.B, self.H, self.Y, self.L)

            loss = loss_W + loss_V + loss_B
            if self.predict_track_params:
                loss_P = self.object_loss(
                    self.W, self.B, self.H, self.P, self.Y, self.L, self.T, self.R
                )
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

    def test_step(self, epoch, thld=0.5):
        self.model.eval()
        losses = {"total": [], "W": [], "V": [], "B": [], "P": [], "acc": []}
        with torch.no_grad():
            for _batch_idx, data in enumerate(self.test_loader):
                data = data.to(self.device)
                if self.predict_track_params:
                    self.W, self.H, self.B, self.P = self.model(
                        data.x, data.edge_index, data.edge_attr
                    )
                else:
                    self.W, self.H, self.B = self.model(
                        data.x, data.edge_index, data.edge_attr
                    )
                self.L = data.particle_id
                self.Y, self.W = data.y, self.W.squeeze(1)
                self.L, self.T = data.particle_id, data.pt
                self.R = data.reconstructable.long()
                self.B = self.B.squeeze()
                loss_W = self.edge_weight_loss(
                    self.W, self.B, self.H, self.Y, self.L
                ).item()
                loss_V = self.potential_loss(
                    self.W, self.B, self.H, self.Y, self.L
                ).item()
                loss_B = self.background_loss(
                    self.W, self.B, self.H, self.Y, self.L
                ).item()
                if self.predict_track_params:
                    loss_P = self.object_loss(
                        self.W, self.B, self.H, self.P, self.Y, self.L, self.T, self.R
                    ).item()
                    losses["P"].append(loss_P)
                acc, TPR, TNR = binary_classification_stats(self.W, self.Y, thld)

                losses["total"].append(loss_W + loss_V + loss_B)
                losses["W"].append(loss_W)
                losses["V"].append(loss_V)
                losses["B"].append(loss_B)
                losses["acc"].append(acc.item())

        losses = {k: np.nanmean(v) for k, v in losses.items()}
        print("test", losses)
        self.test_loss.append(pd.DataFrame(losses, index=[epoch]))

    def validate(self):
        self.model.eval()
        opt_thlds, accs = [], []
        for _batch_idx, data in enumerate(self.val_loader):
            data = data.to(self.device)
            if self.predict_track_params:
                self.W, self.H, self.B, self.P = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
            else:
                self.W, self.H, self.B = self.model(
                    data.x, data.edge_index, data.edge_attr
                )
            self.Y, self.W = data.y, self.W.squeeze(1)
            self.L = data.particle_id
            self.B = self.B.squeeze()
            diff, opt_thld, opt_acc = 100, 0, 0
            for thld in np.arange(0.01, 0.5, 0.01):
                acc, TPR, TNR = binary_classification_stats(self.W, self.Y, thld)
                delta = abs(TPR - TNR)
                if delta < diff:
                    diff, opt_thld, opt_acc = delta, thld, acc
            opt_thlds.append(opt_thld)
            accs.append(opt_acc)
        return np.nanmean(opt_thlds)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"---- Epoch {epoch} ----")
            self.train_step(epoch)
            thld = self.validate()
            self.test_step(epoch, thld=thld)
            # self.scheduler.step()
