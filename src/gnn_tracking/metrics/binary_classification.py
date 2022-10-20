from __future__ import annotations

from functools import cached_property

import torch

from gnn_tracking.utils.types import assert_int


class BinaryClassificationStats:
    def __init__(
        self, output: torch.Tensor, y: torch.Tensor, thld: torch.Tensor | float
    ):
        """

        Args:
            output:
            y:
            thld:

        Returns:
            accuracy, TPR, TNR
        """
        assert_int(y)
        self._output = output
        self._y = y
        self._thld = thld

    @cached_property
    def TP(self) -> float:
        return torch.sum((self._y == 1) & (self._output > self._thld)).item()

    @cached_property
    def TN(self) -> float:
        return torch.sum((self._y == 0) & (self._output < self._thld)).item()

    @cached_property
    def FP(self) -> float:
        return torch.sum((self._y == 0) & (self._output > self._thld)).item()

    @cached_property
    def FN(self) -> float:
        return torch.sum((self._y == 1) & (self._output < self._thld)).item()

    @cached_property
    def acc(self) -> float:
        return zero_divide(self.TP + self.TN, self.TP + self.TN + self.FP + self.FN)

    @cached_property
    def TPR(self) -> float:
        return zero_divide(self.TP, self.TP + self.FN)

    @cached_property
    def TNR(self) -> float:
        return zero_divide(self.TN, self.TN + self.FP)

    @cached_property
    def FPR(self) -> float:
        return zero_divide(self.FP, self.FP + self.TN)

    @cached_property
    def FNR(self) -> float:
        return zero_divide(self.FN, self.FN + self.TP)

    def get_all(self) -> dict[str, float]:
        return {
            "acc": self.acc,
            "TPR": self.TPR,
            "TNR": self.TNR,
            "FPR": self.FPR,
            "FNR": self.FNR,
        }


def zero_divide(a: float, b: float) -> float:
    if b == 0:
        return 0
    return a / b
