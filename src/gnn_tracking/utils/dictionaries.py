import inspect
import itertools
from copy import deepcopy
from typing import Any, Sequence, TypeVar

import torch

_P = TypeVar("_P")


def add_key_prefix(dct: dict[str, _P], prefix: str = "") -> dict[str, _P]:
    """Return a copy of the dictionary with the prefix added to all keys."""
    return {f"{prefix}{k}": v for k, v in dct.items()}


def add_key_suffix(dct: dict[str, _P], suffix: str = "") -> dict[str, _P]:
    """Return a copy of the dictionary with the suffix added to all keys."""
    return {f"{k}{suffix}": v for k, v in dct.items()}


def subdict_with_prefix_stripped(dct: dict[str, _P], prefix: str = "") -> dict[str, _P]:
    """Return a copy of the dictionary for all keys that start with prefix
    and with the prefix removed from all keys."""
    return {k[len(prefix) :]: v for k, v in dct.items() if k.startswith(prefix)}


def expand_grid(
    grid: dict[str, Sequence], fixed: dict[str, Sequence] = None
) -> list[dict[str, Any]]:
    """Expands a grid of parameters into a list of configurations."""
    if fixed is None:
        fixed = {}
    _configs = list(itertools.product(*grid.values()))
    configs: list[dict[str, Any]] = []
    for _c in _configs:
        c = deepcopy(fixed)
        c.update({k: v for k, v in zip(grid.keys(), _c)})
        configs.append(c)
    return configs


def pivot_record_list(records: list[dict]) -> dict:
    """Transform list of key value pairs into dict of lists."""
    keys = list(records[0].keys())
    for record in records:
        if not set(record.keys()) == set(keys):
            raise ValueError("All records must have the same keys.")
    return {k: [r[k] for r in records] for k in keys}


def to_floats(inpt: Any) -> Any:
    """Convert all tensors in a datastructure to floats.
    Works on single tensors, lists, or dictionaries, nested or not.
    """
    if isinstance(inpt, dict):
        return {k: to_floats(v) for k, v in inpt.items()}
    elif isinstance(inpt, list):
        return [to_floats(v) for v in inpt]
    elif isinstance(inpt, torch.Tensor):
        return float(inpt)
    return inpt


def separate_init_kwargs(kwargs: dict, cls: type) -> tuple[dict, dict]:
    cls_argnames = inspect.signature(cls).parameters.keys()
    cls_kwargs = {k: v for k, v in kwargs.items() if k in cls_argnames}
    remaining_kwargs = {k: v for k, v in kwargs.items() if k not in cls_argnames}
    return cls_kwargs, remaining_kwargs
