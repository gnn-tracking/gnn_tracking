from __future__ import annotations

import torch


def zero_divide(a: float, b: float) -> float:
    if b == 0:
        return 0
    return a / b


def binary_classification_stats(
    output: torch.Tensor, y: torch.Tensor, thld: torch.Tensor | float
) -> tuple[float, float, float]:
    """

    Args:
        output:
        y:
        thld:

    Returns:
        accuracy, TPR, TNR
    """
    TP = torch.sum((y == 1) & (output > thld)).item()
    TN = torch.sum((y == 0) & (output < thld)).item()
    FP = torch.sum((y == 0) & (output > thld)).item()
    FN = torch.sum((y == 1) & (output < thld)).item()
    acc = zero_divide(TP + TN, TP + TN + FP + FN)
    TPR = zero_divide(TP, TP + FN)
    TNR = zero_divide(TN, TN + FP)
    return acc, TPR, TNR
