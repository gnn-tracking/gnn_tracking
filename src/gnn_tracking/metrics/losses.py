"""This module contains loss functions for the GNN tracking model."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch.nn.functional import binary_cross_entropy, mse_loss, relu
from typing_extensions import TypeAlias

from gnn_tracking.utils.log import logger

T: TypeAlias = torch.Tensor


# Follows the implementation in kornia at
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
# (binary_focal_loss_with_logits function)
def binary_focal_loss(
    *,
    inpt: T,
    target: T,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    pos_weight: T | None = None,
    mask_outliers=True,
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
        mask = torch.isclose(inpt, torch.Tensor([0.0]).to(inpt.device)) | torch.isclose(
            inpt, torch.Tensor([1.0]).to(inpt.device)
        )
        n_outliers = mask.sum()
        mask = ~mask.bool()
        if n_outliers:
            logger.warning(
                "Masking %d/%d as outliers in focal loss", n_outliers, len(mask)
            )
            logger.debug(inpt[:10])
            logger.debug(target[:10])
    else:
        mask = torch.full_like(inpt, True).bool()

    inpt = inpt[mask]
    target = target[mask]
    pos_weight = pos_weight[mask]

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

    if torch.isnan(loss).any():
        logger.error(
            "NaN loss in focal loss. Here's some more information: "
            "sum pos_term: %s, sum neg_term: %s, sum loss_tmp: %s, "
            "max probs_pos: %s, max probs_neg: %s, max target: %s, ",
            pos_term.sum(),
            neg_term.sum(),
            loss_tmp.sum(),
            probs_pos.max(),
            probs_neg.max(),
            target.max(),
        )

    return loss


def falsify_low_pt_edges(*, y: T, edge_index: T, pt: T, pt_thld: float = 0.0):
    """Modify the ground truth to-be-predicted by the edge classification
    to consider edges that include a hit with pt < pt_thld as false.

    Args:
        y: True classification
        edge_index:
        pt: Hit pt
        pt_thld: Apply pt threshold

    Returns:
        True classification with additional criteria applied
    """
    if math.isclose(pt_thld, 0.0):
        return y
    # Because false edges are already falsified, we can
    # it's enough to check the first hit of the edge for its pt
    return (y.bool() & (pt[edge_index[0, :]] > pt_thld)).long()


class AbstractEdgeWeightLoss(torch.nn.Module, ABC):
    """Abstract base class for loss functions for edge classification."""

    def __init__(self, *, pt_thld: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pt_thld = pt_thld

    def forward(self, *, w: T, y: T, edge_index: T, pt: T, **kwargs) -> T:
        y = falsify_low_pt_edges(
            y=y, edge_index=edge_index, pt=pt, pt_thld=self.pt_thld
        )
        return self._forward(y=y, w=w)

    @abstractmethod
    def _forward(self, *, w: T, y: T, **kwargs) -> T:
        pass


class EdgeWeightBCELoss(AbstractEdgeWeightLoss):
    """Binary Cross Entropy loss function for edge classification"""

    @staticmethod
    def _forward(*, w: T, y: T, **kwargs) -> T:
        bce_loss = binary_cross_entropy(w, y, reduction="mean")
        return bce_loss


class EdgeWeightFocalLoss(AbstractEdgeWeightLoss):
    def __init__(
        self,
        *,
        alpha=0.25,
        gamma=2.0,
        pos_weight=None,
        reduction="mean",
        **kwargs,
    ):
        """Loss function based on focal loss for edge classification.
        See `binary_focal_loss` for details.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def _forward(self, *, w: T, y: T, **kwargs) -> T:
        focal_loss = binary_focal_loss(
            inpt=w,
            target=y,
            alpha=self.alpha,
            gamma=self.gamma,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
            mask_outliers=True,
        )
        return focal_loss


class PotentialLoss(torch.nn.Module):
    def __init__(self, q_min=0.01, radius_threshold=10.0, attr_pt_thld=0.9):
        """Potential/condensation loss (specific to object condensation approach).

        Args:
            q_min: Minimal charge ``q``
            radius_threshold: Parameter of repulsive potential
            attr_pt_thld: Truth-level threshold for hits/tracks to consider in
                attractive loss [GeV]
        """
        super().__init__()
        self.q_min = q_min
        self.radius_threshold = radius_threshold
        self.pt_thld = attr_pt_thld

    def _condensation_loss(
        self, *, beta: T, x: T, particle_id: T, mask: T
    ) -> dict[str, T]:
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
            "attractive": torch.sum(torch.mean(va[mask], dim=0)),
            "repulsive": torch.sum(torch.mean(vr, dim=0)),
        }

    # noinspection PyUnusedLocal
    def forward(
        self,
        *,
        beta: T,
        x: T,
        particle_id: T,
        reconstructable: T,
        track_params: T,
        **kwargs,
    ) -> dict[str, T]:
        mask = (reconstructable > 0) & (track_params > self.pt_thld)
        return self._condensation_loss(
            beta=beta, x=x, particle_id=particle_id, mask=mask
        )


class BackgroundLoss(torch.nn.Module):
    def __init__(self, sb=0.1):
        super().__init__()
        #: Strength of noise suppression
        self.sb = sb

    def _background_loss(self, *, beta: T, particle_id: T) -> T:
        pids = torch.unique(particle_id[particle_id > 0])
        pid_masks = particle_id[:, None] == pids[None, :]
        alphas = torch.argmax(pid_masks * beta[:, None], dim=0)
        beta_alphas = beta[alphas]
        loss = torch.mean(1 - beta_alphas)
        noise_mask = particle_id == 0
        if noise_mask.any():
            loss = loss + self.sb * torch.mean(beta[noise_mask])
        return loss

    # noinspection PyUnusedLocal
    def forward(self, *, beta, particle_id, **kwargs):
        return self._background_loss(beta=beta, particle_id=particle_id)


class ObjectLoss(torch.nn.Module):
    def __init__(self, mode="efficiency"):
        """Loss functions for predicted object properties."""
        super().__init__()
        self.mode = mode

    def _mse(self, *, pred, truth):
        return torch.sum(mse_loss(pred, truth, reduction="none"), dim=1)

    def object_loss(self, *, pred, beta, truth, particle_id):
        # shape: n_nodes
        mse = self._mse(pred=pred, truth=truth)
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

    # noinspection PyUnusedLocal
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
