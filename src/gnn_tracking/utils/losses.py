from __future__ import annotations

import torch
from torch.nn.functional import binary_cross_entropy, mse_loss


class EdgeWeightLoss(torch.nn.Module):
    def forward(self, w, beta, x, y, particle_id):
        bce_loss = binary_cross_entropy(w, y, reduction="mean")
        return bce_loss


class PotentialLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, device="cpu"):
        super(PotentialLoss, self).__init__()
        self.q_min = q_min
        self.device = device

    def V_attractive(self, x, x_alpha, q_alpha):
        norm_sq = torch.norm(x - x_alpha, dim=1) ** 2
        return norm_sq * q_alpha

    def V_repulsive(self, x, x_alpha, q_alpha):
        diffs = 1 - torch.norm(x - x_alpha, dim=1)
        hinges = torch.maximum(torch.zeros(len(x)).to(self.device), diffs)
        return hinges * q_alpha

    def condensation_loss(self, beta, x, particle_id, q_min=1):
        loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        q = torch.arctanh(beta) ** 2 + q_min
        for pid in torch.unique(particle_id):
            p = pid.item()
            if p == 0:
                continue
            M = (particle_id == p).squeeze(-1)
            q_pid = q[M]
            x_pid = x[M]
            M = M.long()
            alpha = torch.argmax(q_pid)
            q_alpha = q_pid[alpha]
            x_alpha = x_pid[alpha]
            va = self.V_attractive(x, x_alpha, q_alpha, device=self.device)
            vr = self.V_repulsive(x, x_alpha, q_alpha, device=self.device)
            loss += torch.mean(q * (M * va + 10 * (1 - M) * vr))
        return loss

    def forward(self, w, beta, x, y, particle_id):
        return self.condensation_loss(beta, x, particle_id, self.q_min)


class BackgroundLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, sb=0.1, device="cpu"):
        super(BackgroundLoss, self).__init__()
        self.sb = sb
        self.q_min = q_min
        self.device = device

    def background_loss(self, beta, x, particle_id, q_min=1, sb=10):
        loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        unique_pids = torch.unique(particle_id[particle_id > 0])
        beta_alphas = torch.zeros(len(unique_pids)).to(self.device)
        for i, pid in enumerate(unique_pids):
            p = pid.item()
            if p == 0:
                continue
            M = (particle_id == p).squeeze(-1)
            beta_pid = beta[M]
            alpha = torch.argmax(beta_pid)
            beta_alpha = beta_pid[alpha]
            beta_alphas[i] = beta_alpha

        n = (particle_id == 0).long()
        nb = torch.sum(n)
        if nb == 0:
            return torch.tensor(0, dtype=float)
        return torch.mean(1 - beta_alphas) + sb * torch.sum(n * beta) / nb

    def forward(self, w, beta, x, y, particle_id):
        return self.background_loss(beta, x, particle_id, self.q_min, self.sb)


class ObjectLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, sb=0.1, device="cpu", mode="efficiency"):
        super(ObjectLoss, self).__init__()
        self.sb = sb
        self.q_min = q_min
        self.device = device
        self.mode = mode

    def MSE(self, p, t):
        return torch.sum(mse_loss(p, t, reduction="none"), dim=1)

    def object_loss(self, pred, beta, truth, particle_id):
        n = (particle_id == 0).long()
        xi = (1 - n) * (torch.arctanh(beta)) ** 2
        mse = self.MSE(pred, truth)
        if self.mode == "purity":
            return 100 / torch.sum(xi) * torch.mean(xi * mse)
        loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        K = torch.tensor(0.0, dtype=torch.float).to(self.device)
        M = torch.tensor(0.0, dtype=torch.long).to(self.device)
        for pid in torch.unique(particle_id):
            p = pid.item()
            if p == 0:
                continue
            M = (particle_id == p).squeeze(-1)
            xi_p = M * p
            weight = 1.0 / (torch.sum(xi_p))
            loss += weight * torch.sum(mse * xi_p)
            K += 1.0
        return 100 * loss / K

    def forward(self, W, beta, H, pred, Y, particle_id, track_params, reconstructable):
        mask = reconstructable > 0
        return self.object_loss(
            pred[mask], beta[mask], track_params[mask], particle_id[mask]
        )
