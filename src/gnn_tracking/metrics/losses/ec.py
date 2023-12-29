import math
from abc import ABC, abstractmethod

import torch
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor as T
from torch.nn.functional import binary_cross_entropy

# ruff: noqa: ARG002


@torch.jit.script
def _binary_focal_loss(
    *,
    inpt: T,
    target: T,
    alpha: float,
    gamma: float,
    pos_weight: T,
) -> T:
    """Extracted function for JIT compilation."""
    probs_pos = inpt
    probs_neg = 1 - inpt

    pos_term = -alpha * pos_weight * probs_neg.pow(gamma) * target * probs_pos.log()
    neg_term = -(1.0 - alpha) * probs_pos.pow(gamma) * (1.0 - target) * probs_neg.log()
    loss_tmp = pos_term + neg_term

    return torch.mean(loss_tmp)


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
    if pos_weight is None:
        pos_weight = torch.tensor([1.0], device=inpt.device)

    assert gamma >= 0.0
    assert 0 <= alpha <= 1
    assert not torch.isnan(inpt).any()
    assert not torch.isnan(target).any()
    assert pos_weight is not None

    # JIT compilation does not support optional arguments
    return _binary_focal_loss(
        inpt=inpt,
        target=target,
        alpha=alpha,
        gamma=gamma,
        pos_weight=pos_weight,
    )


def falsify_low_pt_edges(
    *, y: T, edge_index: T | None = None, pt: T | None = None, pt_thld: float = 0.0
) -> T:
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
    assert edge_index is not None
    assert pt is not None
    # Because false edges are already falsified, we can
    # it's enough to check the first hit of the edge for its pt
    return y.bool() & (pt[edge_index[0, :]] > pt_thld)


class FalsifyLowPtEdgeWeightLoss(torch.nn.Module, ABC, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(self, *, pt_thld: float = 0.0):
        """Add an option to falsify edges with low pt to edge classification losses."""
        super().__init__()
        self.save_hyperparameters()

    # noinspection PyUnusedLocal
    def forward(
        self, *, w: T, y: T, edge_index: T | None = None, pt: T | None = None, **kwargs
    ) -> T:
        y = falsify_low_pt_edges(
            y=y, edge_index=edge_index, pt=pt, pt_thld=self.hparams.pt_thld
        )
        return self._forward(y=y.float(), w=w)

    @abstractmethod
    def _forward(self, *, w: T, y: T, **kwargs) -> T:
        pass


class EdgeWeightBCELoss(FalsifyLowPtEdgeWeightLoss):
    """Binary Cross Entropy loss function for edge classification"""

    @staticmethod
    def _forward(*, w: T, y: T, **kwargs) -> T:
        return binary_cross_entropy(w, y, reduction="mean")


class EdgeWeightFocalLoss(FalsifyLowPtEdgeWeightLoss):
    # noinspection PyUnusedLocal
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
        self.save_hyperparameters()

    def _forward(self, *, w: T, y: T, **kwargs) -> T:
        pos_weight = self.hparams.pos_weight
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], device=w.device)
        return binary_focal_loss(
            inpt=w,
            target=y,
            alpha=self.hparams.alpha,
            gamma=self.hparams.gamma,
            pos_weight=pos_weight,
        )


class HaughtyFocalLoss(torch.nn.Module, HyperparametersMixin):
    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pt_thld=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._alpha = alpha
        self._gamma = gamma
        self._pt_thld = pt_thld

    # noinspection PyUnusedLocal
    def forward(self, *, w: T, y: T, edge_index: T, pt: T, **kwargs) -> T:
        pos_weight = falsify_low_pt_edges(
            y=y, edge_index=edge_index, pt=pt, pt_thld=self._pt_thld
        )
        return binary_focal_loss(
            inpt=w,
            target=y.long(),
            alpha=self._alpha,
            gamma=self._gamma,
            pos_weight=pos_weight,
        )
