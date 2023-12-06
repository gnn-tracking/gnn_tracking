"""This module contains loss functions for the GNN tracking model."""

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Union

import torch
from torch import Tensor as T

from gnn_tracking.utils.log import logger


def unpack_loss_returns(key: str, returns: Any) -> dict[str, float | T]:
    """Some of our loss functions return a dictionary or a list of individual losses.
    This function unpacks these into a dictionary of individual losses with appropriate
    keys.

    Args:
        key: str (name of the loss function)
        returns: dict or list or single value

    Returns:
        dict of individual losses
    """
    if isinstance(returns, Mapping):
        return {f"{key}_{k}": v for k, v in returns.items()}
    if isinstance(returns, (list, tuple)):
        # Don't put 'Sequence' here, because Ts are Sequences
        return {f"{key}_{i}": v for i, v in enumerate(returns)}
    return {key: returns}


@dataclass(kw_only=True)
class MultiLossFctReturn:
    """Return type for loss functions that return multiple losses."""

    #: Split up losses
    loss_dct: dict[str, T]
    #: Weights
    weight_dct: dict[str, T]
    #: Other things that should be logged
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.loss_dct.keys() == self.weight_dct.keys()

    @property
    def loss(self) -> T:
        loss = sum(self.weighted_losses.values())
        assert isinstance(loss, torch.Tensor)
        return loss

    @property
    def weighted_losses(self) -> dict[str, T]:
        return {k: v * self.weight_dct[k] for k, v in self.loss_dct.items()}


class MultiLossFct(torch.nn.Module):
    """Base class for loss functions that return multiple losses."""

    def forward(self, *args: Any, **kwargs: Any) -> MultiLossFctReturn:
        ...


loss_weight_type = Union[float, dict[str, float], list[float]]


class LossClones(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module, prefixes=("w", "y")) -> None:
        """Wrapper for a loss function that evaluates it on multiple inputs.
        The forward method will look for all model outputs that start with `w_`
        (or another specified prefix) and evaluate the loss function for each of them,
        returning a dictionary of losses (with keys equal to the suffixes).

        Usage example 1:

        .. code-block:: python

            losses = {
                "potential": (PotentialLoss(), 1.),
                "edge": (LossClones(EdgeWeightBCELoss()), [1.0, 2.0, 3.0])
            }

        will evaluate three clones of the BCE loss function, one for each EC layer.

        Usage Example 2:


        .. code-block:: python

            losses = {
                "potential": (PotentialLoss(), 1.),
                "edge": (LossClones(EdgeWeightBCELoss()), {}))
            }

        this works with a variable number of layers. The weights are all 1.

        Under the hood, ``ECLossClones(EdgeWeightBCELoss())(model output)`` will output
        a dictionary of the individual losses, keyed by their suffixes (in a similar
        way to how `PotentialLoss` returns a dictionary of losses).

        Args:
            loss: Loss function to be evaluated on multiple inputs
            prefixes: Prefixes of the model outputs that should be evaluated.
                An underscore is assumed (set prefix to `w` for `w_0`, `w_1`, etc.)

        """
        super().__init__()
        self._loss = loss
        self._prefixes = prefixes

    def forward(self, **kwargs) -> dict[str, T]:
        kwargs = copy.copy(kwargs)
        for prefix in self._prefixes:
            if prefix in kwargs:
                logger.warning(
                    f"LossClones prefix {prefix} is also a model output. Removing "
                    f"this for now, but you probably want to clean up if this is not "
                    f"intended."
                )
                kwargs.pop(prefix)
        losses = {}
        ec_layer_names = sorted(
            [
                k[len(self._prefixes[0]) + 1 :]
                for k in kwargs
                if k.startswith(self._prefixes[0] + "_")
            ]
        )
        for layer_name in ec_layer_names:
            rename_dct = {f"{prefix}_{layer_name}": prefix for prefix in self._prefixes}
            renamed_kwargs = {rename_dct.get(k, k): v for k, v in kwargs.items()}
            loss = self._loss(**renamed_kwargs)
            losses[layer_name] = loss
        return losses
