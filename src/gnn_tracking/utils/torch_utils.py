"""Utility functions for general torch stuff."""

from torch import nn


def freeze(model: nn.Module) -> nn.Module:
    """Freezes all parameters of a model.

    Returns:
        The model with all parameters frozen (but model is also modified in-place).
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def freeze_if(model: nn.Module | None, do_freeze: bool = False) -> nn.Module | None:
    """Freezes all parameters of a model if `do_freeze` is True. If model is None,
    None is returned. This is a trivial convenience function to avoid if-else
    statements.

    Returns:
        The model with all parameters frozen (but model is also modified in-place).
    """
    if model is None:
        return None
    if do_freeze:
        return freeze(model)
    return model
