from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import DefaultDict

import numpy as np

from gnn_tracking.utils.log import logger


class DynamicLossWeights(ABC):
    def __init__(self, **kwargs):
        """Abstract base class to implement dynamic loss weights, i.e. weights for
        the different loss functions that change during the training.
        """
        self._current_loss_weights: DefaultDict[str, float] = collections.defaultdict(
            lambda: 1.0
        )

    def get(self) -> DefaultDict[str, float]:
        """Get loss weights"""
        return self._current_loss_weights

    def __getitem__(self, item: str) -> float:
        """Get loss weight for specific loss function"""
        return self.get()[item]

    def step(self, losses: dict[str, float]) -> DefaultDict[str, float]:
        """Will be called after each epoch.

        Args:
            losses: Losses of the current epoch.

        Returns:
            New loss weights
        """
        self._current_loss_weights.update(self._step(losses))
        return self.get()

    @abstractmethod
    def _step(self, losses: dict[str, float]) -> DefaultDict[str, float]:
        pass


class ConstantLossWeights(DynamicLossWeights):
    def __init__(self, loss_weights: dict[str, float] | None = None):
        """Constant loss weights."""
        super().__init__()
        if loss_weights is not None:
            self._current_loss_weights.update(loss_weights)

    def _step(self, losses: dict[str, float]) -> DefaultDict[str, float]:
        return self._current_loss_weights


class NormalizeAt(DynamicLossWeights):
    def __init__(
        self,
        at: list,
        relative_weights: list[dict[str, float]] | None = None,
        rolling=5,
    ):
        """Normalize losses at specific epochs.

        Args:
            at: Normalize at these epochs (starting count from 0)
            relative_weights: Relative weights between different losses: None (all
                equal), or list of equal length as ``at``. Each list item dictionary
                with relative weights. Any key that is not present will be set to 1.
            rolling: Use this many last losses for normalization
        """
        super().__init__()
        self.at = at
        if relative_weights is not None and not len(relative_weights) == len(at):
            raise ValueError("Length of relative_weights must match length of at")
        self.relative_weights = relative_weights
        self._n_epoch = 0
        self._losses: DefaultDict[str, list[float]] = collections.defaultdict(list)
        self._rolling = rolling

    def _get_relative_weights(self, key: str) -> float:
        if self.relative_weights is None:
            return 1.0
        return self.relative_weights[self.at.index(self._n_epoch)].get(key, 1.0)

    def _get_scaling_config(self, key: str) -> float:
        v = np.abs(np.mean(self._losses[key][-self._rolling :]))
        if np.isclose(v, 0.0):
            logger.warning("Loss is 0, not scaling")
            v = 1
        scale = self._get_relative_weights(key) / v
        assert scale > 0
        if np.isnan(scale):
            logger.critical("Scaling is NaN!")
        return scale

    def _step(self, losses: dict[str, float]) -> DefaultDict[str, float]:
        for k, v in losses.items():
            self._losses[k].append(v)
        if self._n_epoch in self.at:
            self._current_loss_weights.update(
                {k: self._get_scaling_config(k) for k in self._losses}
            )
        self._n_epoch += 1
        return self._current_loss_weights
