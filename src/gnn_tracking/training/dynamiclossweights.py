from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import DefaultDict


class DynamicLossWeights(ABC):
    def __init__(self, **kwargs):
        self._current_loss_weights = {}

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
        self.loss_weights: DefaultDict[str, float] = collections.defaultdict(
            lambda: 1.0
        )
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

    def _step(self, losses: dict[str, float]) -> dict[str, float]:
        return self.loss_weights
