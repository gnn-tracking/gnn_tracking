"""This module contains loss functions for the GNN tracking model."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy, mse_loss, relu
from typing_extensions import TypeAlias

from gnn_tracking.utils.log import logger

T: TypeAlias = torch.Tensor


@torch.jit.script
def _binary_focal_loss(
    *,
    inpt: T,
    target: T,
    mask: T,
    alpha: float,
    gamma: float,
    pos_weight: T,
) -> T:
    """Extracted function for JIT compilation."""
    inpt = inpt[mask]
    target = target[mask]
    pos_weight = pos_weight[mask]

    probs_pos = inpt
    probs_neg = 1 - inpt

    pos_term = -alpha * pos_weight * probs_neg.pow(gamma) * target * probs_pos.log()
    neg_term = -(1 - alpha) * probs_pos.pow(gamma) * (1.0 - target) * probs_neg.log()
    loss_tmp = pos_term + neg_term

    loss = torch.mean(loss_tmp)

    return loss


# Follows the implementation in kornia at
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
# (binary_focal_loss_with_logits function)
def binary_focal_loss(
    *,
    inpt: T,
    target: T,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: T | None = None,
) -> T:
    """Binary Focal Loss, following https://arxiv.org/abs/1708.02002.

    Args:
        inpt:
        target:
        alpha: Weight for positive/negative results
        gamma: Focusing parameter
        pos_weight: Can be used to balance precision/recall
    """
    assert gamma >= 0.0
    assert 0 <= alpha <= 1
    assert not torch.isnan(inpt).any()
    assert not torch.isnan(target).any()

    if pos_weight is None:
        pos_weight = torch.ones(inpt.shape[-1], device=inpt.device, dtype=inpt.dtype)

    # Masking outliers
    mask = ~(
        torch.isclose(inpt, torch.Tensor([0.0]).to(inpt.device))
        | torch.isclose(inpt, torch.Tensor([1.0]).to(inpt.device))
    ).bool()
    if not mask.all():
        logger.warning(
            "Masking %d/%d as outliers in focal loss", (~mask).sum(), len(mask)
        )

    return _binary_focal_loss(
        inpt=inpt,
        target=target,
        mask=mask,
        alpha=alpha,
        gamma=gamma,
        pos_weight=pos_weight,
    )


def falsify_low_pt_edges(*, y: T, edge_index: T, pt: T, pt_thld: float = 0.0) -> T:
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


class FalsifyLowPtEdgeWeightLoss(torch.nn.Module, ABC):
    def __init__(self, *, pt_thld: float = 0.0, **kwargs):
        """Add an option to falsify edges with low pt to edge classification losses."""
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


class EdgeWeightBCELoss(FalsifyLowPtEdgeWeightLoss):
    """Binary Cross Entropy loss function for edge classification"""

    @staticmethod
    def _forward(*, w: T, y: T, **kwargs) -> T:
        bce_loss = binary_cross_entropy(w, y, reduction="mean")
        return bce_loss


class EdgeWeightFocalLoss(FalsifyLowPtEdgeWeightLoss):
    def __init__(
        self,
        *,
        alpha=0.25,
        gamma=2.0,
        pos_weight=None,
        **kwargs,
    ):
        """Loss function based on focal loss for edge classification.
        See `binary_focal_loss` for details.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def _forward(self, *, w: T, y: T, **kwargs) -> T:
        focal_loss = binary_focal_loss(
            inpt=w,
            target=y,
            alpha=self.alpha,
            gamma=self.gamma,
            pos_weight=self.pos_weight,
        )
        return focal_loss


class HaughtyFocalLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pt_thld=0.0,
    ):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._pt_thld = pt_thld

    # noinspection PyUnusedLocal
    def forward(self, *, w: T, y: T, edge_index: T, pt: T, **kwargs) -> T:
        pos_weight = falsify_low_pt_edges(
            y=y, edge_index=edge_index, pt=pt, pt_thld=self._pt_thld
        )
        focal_loss = binary_focal_loss(
            inpt=w,
            target=y,
            alpha=self._alpha,
            gamma=self._gamma,
            pos_weight=pos_weight,
        )
        return focal_loss


@torch.jit.script
def _condensation_loss(
    *,
    beta: T,
    x: T,
    particle_id: T,
    mask: T,
    q_min: float,
    radius_threshold: float,
) -> dict[str, T]:
    """Extracted function for JIT-compilation. See `PotentialLoss` for details."""
    pids = torch.unique(particle_id[particle_id > 0])
    # n_nodes x n_pids
    pid_masks = particle_id[:, None] == pids[None, :]  # type: ignore

    q = torch.arctanh(beta) ** 2 + q_min
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
        * (relu(radius_threshold - torch.sqrt(norm_sq + 1e-8)) * q_alphas).squeeze(
            dim=0
        )
    )

    return {
        "attractive": torch.sum(torch.mean(va[mask], dim=0)),
        "repulsive": torch.sum(torch.mean(vr, dim=0)),
    }


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

    # noinspection PyUnusedLocal
    def forward(
        self,
        *,
        beta: T,
        x: T,
        particle_id: T,
        reconstructable: T,
        track_params: T,
        ec_hit_mask: T,
        **kwargs,
    ) -> dict[str, T]:
        # If a post-EC node mask was applied in the model, then all model outputs
        # already include this mask, while everything gotten from the data
        # does not. Hence, we apply it here.
        particle_id = particle_id[ec_hit_mask]
        reconstructable = reconstructable[ec_hit_mask]
        track_params = track_params[ec_hit_mask]
        mask = (reconstructable > 0) & (track_params > self.pt_thld)
        return _condensation_loss(
            beta=beta,
            x=x,
            particle_id=particle_id,
            mask=mask,
            q_min=self.q_min,
            radius_threshold=self.radius_threshold,
        )


@torch.jit.script
def _background_loss(*, beta: T, particle_id: T, sb: float) -> T:
    """Extracted function for JIT-compilation. See `BackgroundLoss` for details."""
    pids = torch.unique(particle_id[particle_id > 0])
    pid_masks = particle_id[:, None] == pids[None, :]
    alphas = torch.argmax(pid_masks * beta[:, None], dim=0)
    beta_alphas = beta[alphas]
    loss = torch.mean(1 - beta_alphas)
    noise_mask = particle_id == 0
    if noise_mask.any():
        loss = loss + sb * torch.mean(beta[noise_mask])
    return loss


class BackgroundLoss(torch.nn.Module):
    def __init__(self, sb=0.1):
        super().__init__()
        #: Strength of noise suppression
        self.sb = sb

    # noinspection PyUnusedLocal
    def forward(self, *, beta: T, particle_id: T, ec_hit_mask: T, **kwargs) -> T:
        return _background_loss(
            beta=beta, particle_id=particle_id[ec_hit_mask], sb=self.sb
        )


class ObjectLoss(torch.nn.Module):
    def __init__(self, mode="efficiency"):
        """Loss functions for predicted object properties."""
        super().__init__()
        self.mode = mode

    def _mse(self, *, pred: T, truth: T) -> T:
        return torch.sum(mse_loss(pred, truth, reduction="none"), dim=1)

    def object_loss(self, *, pred: T, beta: T, truth: T, particle_id: T) -> T:
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
        self,
        *,
        beta: T,
        pred: T,
        particle_id: T,
        track_params: T,
        reconstructable: T,
        **kwargs,
    ) -> T:
        mask = reconstructable > 0
        return self.object_loss(
            pred=pred[mask],
            beta=beta[mask],
            truth=track_params[mask],
            particle_id=particle_id[mask],
        )


class LossFctType(Protocol):
    """Type of a loss function"""

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        ...

    def to(self, device: torch.device) -> LossFctType:
        ...
