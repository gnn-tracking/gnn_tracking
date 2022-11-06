from __future__ import annotations

import functools
import inspect
from typing import Any, Callable


def get_all_argument_names(func: Callable) -> list[str]:
    """Return all argument names of function"""
    sig = inspect.signature(func)
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def remove_irrelevant_arguments(
    func: Callable, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Remove all keys from ``kwargs`` that are not a named argument for
    ``func``.
    """
    return {k: v for k, v in kwargs.items() if k in get_all_argument_names(func)}


def tolerate_additional_kwargs(func: Callable) -> Callable:
    """A decorator to make a function accept (and ignore) additional keyword
    arguments.
    """

    @functools.wraps(func)
    def wrapped(**kwargs):
        return func(**remove_irrelevant_arguments(func, kwargs))

    return wrapped
