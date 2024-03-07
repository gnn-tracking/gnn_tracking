"""This class collects figures of merit for binary classification"""

from functools import cached_property
from typing import Iterable

import numpy as np
import torch
from torchmetrics.classification import BinaryAUROC

from gnn_tracking.utils.device import guess_device
from gnn_tracking.utils.log import logger


class BinaryClassificationStats:
    def __init__(
        self, output: torch.Tensor, y: torch.Tensor, thld: torch.Tensor | float
    ):
        """Calculator for binary classification metrics.
        All properties are cached, so they are only calculated once.

        Args:
            output: Output weights
            y: True labels
            thld: Threshold to consider something true

        Returns:
            accuracy, TPR, TNR
        """
        self._output = output
        self._y = y.int()
        self._thld = thld

    @cached_property
    def _true(self) -> torch.Tensor:
        return self._y == 1

    @cached_property
    def n_true(self) -> int:
        return torch.sum(self._true).item()

    @cached_property
    def _false(self) -> torch.Tensor:
        return ~self._true

    @cached_property
    def n_false(self) -> int:
        return len(self._y) - self.n_true

    @cached_property
    def _predicted_false(self) -> torch.Tensor:
        return self._output < self._thld

    @cached_property
    def n_predicted_false(self) -> int:
        return torch.sum(self._predicted_false).item()

    @cached_property
    def _predicted_true(self) -> torch.Tensor:
        return ~self._predicted_false

    @cached_property
    def n_predicted_true(self) -> int:
        return len(self._y) - self.n_predicted_false

    @cached_property
    def TP(self) -> float:
        return torch.sum(self._true & self._predicted_true).item()

    @cached_property
    def TN(self) -> float:
        return torch.sum(self._false & self._predicted_false).item()

    @cached_property
    def FP(self) -> float:
        return torch.sum(self._false & self._predicted_true).item()

    @cached_property
    def FN(self) -> float:
        return torch.sum(self._true & self._predicted_false).item()

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

    @cached_property
    def balanced_acc(self) -> float:
        return (self.TPR + self.TNR) / 2

    @cached_property
    def F1(self) -> float:
        return zero_divide(2 * self.TP, 2 * self.TP + self.FP + self.FN)

    @cached_property
    def MCC(self) -> float:
        return zero_divide(
            self.TP * self.TN - self.FP * self.FN,
            np.sqrt(
                float(
                    (self.TP + self.FP)
                    * (self.TP + self.FN)
                    * (self.TN + self.FP)
                    * (self.TN + self.FN)
                )
            ),
        )

    def get_all(self) -> dict[str, float]:
        return {
            "acc": self.acc,
            "TPR": self.TPR,
            "TNR": self.TNR,
            "FPR": self.FPR,
            "FNR": self.FNR,
            "balanced_acc": self.balanced_acc,
            "F1": self.F1,
            "MCC": self.MCC,
            "n_true": self.n_true,
            "n_false": self.n_false,
            "n_predicted_true": self.n_predicted_true,
            "n_predicted_false": self.n_predicted_false,
        }


def zero_divide(a: float, b: float) -> float:
    """Normal division a/b but return 0 for x/0"""
    if b == 0:
        return 0
    return a / b


def get_maximized_bcs(
    *, output: torch.Tensor, y: torch.Tensor, n_samples=200
) -> dict[str, float]:
    """Calculate the best possible binary classification stats for a given output and y.

    Args:
        output: Weights
        y: True
        n_samples: Number of thresholds to sample

    Returns:
        Dictionary of metrics
    """
    thlds = torch.linspace(0.0, 1.0, n_samples)

    def getter(bcs: BinaryClassificationStats):
        return bcs.balanced_acc, bcs.F1, bcs.TPR, bcs.TNR, bcs.MCC

    results = torch.asarray(
        [
            getter(BinaryClassificationStats(y=y, output=output, thld=thld))
            for thld in thlds
        ],
        device=output.device,
    ).T

    assert results.shape[0] == 5

    bas = results[0, :]
    f1s = results[1, :]
    tprs = results[2, :]
    tnrs = results[3, :]
    mccs = results[4, :]
    r_diff = torch.abs(tprs - tnrs)
    min_diff_idx = torch.argmin(r_diff)
    tpr_eq_tnr = (tprs[min_diff_idx] + tnrs[min_diff_idx]) / 2

    def add_max_and_max_at(dct, key, vals: torch.Tensor) -> None:
        max_idx = torch.argmax(vals)
        dct[key] = vals[max_idx].item()
        dct[f"{key}_loc"] = thlds[max_idx].item()

    dct = {}
    add_max_and_max_at(dct, "max_ba", bas)
    add_max_and_max_at(dct, "max_f1", f1s)
    add_max_and_max_at(dct, "max_mcc", mccs)
    dct["tpr_eq_tnr"] = tpr_eq_tnr.item()
    dct["tpr_eq_tnr_loc"] = thlds[min_diff_idx].item()
    return dct


def roc_auc_score(
    *,
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    max_fpr: float | None = None,
    device=None,
) -> float:
    """Wrapper that ignores exceptions
    that can e.g., be raised if there's only one label present.
    """
    metric = BinaryAUROC(max_fpr=max_fpr).to(device=guess_device(device))
    try:
        return metric(preds=y_score, target=y_true).item()
    except (ValueError, IndexError) as e:
        # Index error is for https://github.com/Lightning-AI/torchmetrics/issues/1891
        logger.error(e)
        return float("nan")


def get_roc_auc_scores(true, predicted, max_fprs: Iterable[float | None]):
    """Calculate ROC AUC scores for a given set of maximum FPRs."""
    metrics = {}
    if None in max_fprs:
        metrics["roc_auc"] = roc_auc_score(y_true=true, y_score=predicted)
    for max_fpr in max_fprs:
        if max_fpr is None:
            continue
        metrics[f"roc_auc_{max_fpr}FPR"] = roc_auc_score(
            y_true=true,
            y_score=predicted,
            max_fpr=max_fpr,
        )
    return metrics


# Also add this to EC metrics
# | get_maximized_bcs(output=predicted, y=true)


# def get_binary_classification_metrics_at_thld(
#     *, edge_index: T, pt: T, w: T, y: T, pt_min: float, thld: float
# ) -> dict[str, float]:
#     """Evaluate edge classification metrics for a given pt threshold and
#     EC threshold.
#
#     Args:
#         pt_min: pt threshold: We discard all edges where both nodes have
#             `pt <= pt_min` before evaluating any metric.
#         thld: EC threshold
#
#     Returns:
#         Dictionary of metrics
#     """
#     pt_a = pt[edge_index[0]]
#     pt_b = pt[edge_index[1]]
#     edge_pt_mask = (pt_a > pt_min) | (pt_b > pt_min)
#
#     predicted = w[edge_pt_mask]
#     true = y[edge_pt_mask].long()
#
#     bcs = BinaryClassificationStats(
#         output=predicted,
#         y=true,
#         thld=thld,
#     )
#     metrics = bcs.get_all()
#     return {denote_pt(k, pt_min): v for k, v in metrics.items()}
#
