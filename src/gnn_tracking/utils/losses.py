from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.nn.functional import binary_cross_entropy, mse_loss, relu

from gnn_tracking.utils.log import logger

T = torch.Tensor


# Follows the implementation in kornia at
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
# (binary_focal_loss_with_logits function)
def binary_focal_loss(
    inpt: T,
    target: T,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    pos_weight: T | None = None,
    mask_outliers=False,
) -> T:
    """Binary Focal Loss, following https://arxiv.org/abs/1708.02002.

    Args:
        inpt:
        target:
        alpha: Weight for positive/negative results
        gamma: Focusing parameter
        reduction: 'none', 'mean', 'sum'
        pos_weight: Can be used to balance precision/recall
        mask_outliers: Mask 0s and 1s in input.
    """
    assert gamma >= 0.0
    assert 0 <= alpha <= 1

    if pos_weight is None:
        pos_weight = torch.ones(inpt.shape[-1], device=inpt.device, dtype=inpt.dtype)

    if mask_outliers:
        mask = torch.isclose(inpt, torch.Tensor(0.0)) | torch.isclose(
            inpt, torch.Tensor(1.0)
        )
        mask = mask.bool()
        n_outliers = mask.sum()
        if n_outliers:
            logger.warning("Masking %d outliers in focal loss", n_outliers)
    else:
        mask = torch.full_like(inpt, True).bool()

    inpt = inpt[mask]
    target = target[mask]

    probs_pos = inpt
    probs_neg = 1 - inpt

    pos_term = -alpha * pos_weight * probs_neg.pow(gamma) * target * probs_pos.log()
    neg_term = -(1 - alpha) * probs_pos.pow(gamma) * (1.0 - target) * probs_neg.log()
    loss_tmp = pos_term + neg_term

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss


class EdgeWeightLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, w, y, **kwargs) -> T:
        pass


class EdgeWeightBCELoss(EdgeWeightLoss):
    @staticmethod
    def forward(*, w, y, **kwargs) -> T:
        bce_loss = binary_cross_entropy(w, y, reduction="mean")
        return bce_loss


class EdgeWeightFocalLoss(EdgeWeightLoss):
    def __init__(self, *, alpha=0.25, gamma=2.0, pos_weight=None, reduction="mean"):
        """See binary_focal_loss for details."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, *, w, y, **kwargs) -> T:
        focal_loss = binary_focal_loss(
            w,
            y,
            alpha=self.alpha,
            gamma=self.gamma,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        return focal_loss


class PotentialLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, radius_threshold=10.0):
        super().__init__()
        self.q_min = q_min
        self.radius_threshold = radius_threshold

    def condensation_loss(self, *, beta: T, x: T, particle_id: T) -> dict[str, T]:
        pids = torch.unique(particle_id[particle_id > 0])
        # n_nodes x n_pids
        pid_masks = particle_id[:, None] == pids[None, :]  # type: ignore

        q = torch.arctanh(beta) ** 2 + self.q_min
        alphas = torch.argmax(q[:, None] * pid_masks, dim=0)
        x_alphas = x[alphas].transpose(0, 1)
        q_alphas = q[alphas][None, None, :]

        diff = x[:, :, None] - x_alphas[None, :, :]
        norm_sq = torch.sum(diff**2, dim=1)

        # Attractive potential
        va = q[:, None] * pid_masks * (norm_sq * q_alphas).squeeze(dim=0)
        # Repulsive potential
        vr = (
            q[:, None]
            * (~pid_masks)
            * (
                relu(self.radius_threshold - torch.sqrt(norm_sq + 1e-8)) * q_alphas
            ).squeeze(dim=0)
        )

        return {
            "attractive": torch.sum(torch.mean(va, dim=0)),
            "repulsive": torch.sum(torch.mean(vr, dim=0)),
        }

    # noinspection PyUnusedVariable
    def forward(self, *, beta: T, x: T, particle_id: T, **kwargs) -> dict[str, T]:
        return self.condensation_loss(beta=beta, x=x, particle_id=particle_id)


class BackgroundLoss(torch.nn.Module):
    def __init__(self, sb=0.1):
        super().__init__()
        #: Strength of noise suppression
        self.sb = sb

    def background_loss(self, *, beta: T, particle_id: T) -> T:
        pids = torch.unique(particle_id[particle_id > 0])
        pid_masks = particle_id[:, None] == pids[None, :]
        alphas = torch.argmax(pid_masks * beta[:, None], dim=0)
        beta_alphas = beta[alphas]
        loss = torch.mean(1 - beta_alphas)
        noise_mask = particle_id == 0
        if noise_mask.any():
            loss = loss + self.sb * torch.mean(beta[noise_mask])
        return loss

    def forward(self, *, beta, particle_id, **kwargs):
        return self.background_loss(beta=beta, particle_id=particle_id)


class ObjectLoss(torch.nn.Module):
    def __init__(self, mode="efficiency"):
        super().__init__()
        #: Strength of noise suppression
        self.mode = mode

    def MSE(self, *, pred, truth):
        return torch.sum(mse_loss(pred, truth, reduction="none"), dim=1)

    def object_loss(self, *, pred, beta, truth, particle_id):
        # shape: n_nodes
        mse = self.MSE(pred=pred, truth=truth)
        if self.mode == "purity":
            noise_mask = particle_id == 0
            # shape: n_nodes
            xi = (~noise_mask) * torch.arctanh(beta) ** 2
            return 1 / torch.sum(xi) * torch.mean(xi * mse)
        elif self.mode == "efficiency":
            # shape: n_pids
            pids = torch.unique(particle_id[particle_id > 0])
            # PID masks (n_nodes x n_pids)
            pid_masks = particle_id[:, None] == pids[None, :]
            # shape: (n_nodes x n_pids)
            xi_p = pid_masks * (torch.arctanh(beta) ** 2)[:, None]
            # shape: n_pids
            xi_p_norm = torch.sum(xi_p, dim=0)
            # shape: n_pids
            terms = torch.sum(mse[:, None] * xi_p, dim=0)
            loss = torch.mean(terms / xi_p_norm)
            return loss
        else:
            raise ValueError("Unknown mode: {mode}")

    # noinspection PyUnusedVariable
    def forward(
        self, *, beta, pred, particle_id, track_params, reconstructable, **kwargs
    ):
        mask = reconstructable > 0
        return self.object_loss(
            pred=pred[mask],
            beta=beta[mask],
            truth=track_params[mask],
            particle_id=particle_id[mask],
        )
