from __future__ import annotations

import torch


def zero_divide(a, b):
    if b == 0:
        return 0
    return a / b


def binary_classification_stats(output, y, thld) -> tuple[float, float, float]:
    """

    Args:
        output:
        y:
        thld:

    Returns:
        accuracy, TPR, TNR
    """
    TP = torch.sum((y == 1) & (output > thld))
    TN = torch.sum((y == 0) & (output < thld))
    FP = torch.sum((y == 0) & (output > thld))
    FN = torch.sum((y == 1) & (output < thld))
    acc = zero_divide(TP + TN, TP + TN + FP + FN)
    TPR = zero_divide(TP, TP + FN)
    TNR = zero_divide(TN, TN + FP)
    return acc, TPR, TNR
