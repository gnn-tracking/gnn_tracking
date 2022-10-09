from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import DefaultDict

import numpy as np

from gnn_tracking.utils.log import logger


class DynamicLossWeights(ABC):
    def __init__(self, **kwargs):
        self._current_loss_weights: DefaultDict[str, float] = collections.defaultdict(
            lambda: 1.0
        )

    def get(self) -> dict[str, float]:
        return self._current_loss_weights

    def __getitem__(self, item):
        return self.get()[item]

    def step(self, losses: dict[str, float]) -> dict[str, float]:
        """Will be called after each epoch.

        Args:
            losses: Losses of the current epoch.

        Returns:
            New loss weights
        """
        self._current_loss_weights = self._step(losses)
        return self.get()

    @abstractmethod
    def _step(self, losses: dict[str, float]) -> dict[str, float]:
        pass


class ConstantLossWeights(DynamicLossWeights):
    def __init__(self, loss_weights: dict[str, float] | None = None):
        super().__init__()
        if loss_weights is not None:
            self._current_loss_weights.update(loss_weights)

    def _step(self, losses: dict[str, float]) -> dict[str, float]:
        return self._current_loss_weights


class NormalizeEvery(DynamicLossWeights):
    def __init__(
        self,
        every=5,
        warmup=1,
        initial_weights: dict[str, float] | None = None,
        rolling=5,
    ):
        """

        Args:
            every: Normalize every this many epochs
            warmup: Return default values for this many epcohs.
            initial_weights: Initial weights for first iteration
            rolling: Use this many last losses for normalization
        """
        super().__init__()
        self.every = every
        if warmup < 1:
            raise ValueError("At least one warmup epoch required!")
        self.warmup = warmup
        if initial_weights is not None:
            self._current_loss_weights.update(initial_weights)
        self._n_epoch = 0
        self._losses = collections.defaultdict(list)
        self._rolling = rolling

    def _get_scaling_config(self, losses: np.ndarray) -> float:
        v = np.abs(np.mean(losses[-self._rolling :]))
        if np.isclose(v, 0.0):
            logger.warning("Loss is 0, not scaling")
            v = 1
        return 1 / v

    def _step(self, losses: dict[str, float]) -> dict[str, float]:
        self._n_epoch += 1
        for k, v in losses.items():
            self._losses[k].append(v)
        if self._n_epoch <= self.warmup:
            return self._current_loss_weights
        if self._n_epoch % self.every == 0:
            self._current_loss_weights.update(
                {
                    k: self._get_scaling_config(np.array(vs))
                    for k, vs in self._losses.items()
                }
            )
        return self._current_loss_weights
