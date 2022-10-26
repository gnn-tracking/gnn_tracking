from __future__ import annotations

from typing import TypeVar

_P = TypeVar("_P")


def add_key_prefix(dct: dict[str, _P], prefix: str = "") -> dict[str, _P]:
    """Return a copy of the dictionary with the prefix added to all keys."""
    return {f"{prefix}{k}": v for k, v in dct.items()}


def subdict_with_prefix_stripped(dct: dict[str, _P], prefix: str = "") -> dict[str, _P]:
    """Return a copy of the dictionary for all keys that start with prefix
    and with the prefix removed from all keys."""
    return {k[len(prefix) :]: v for k, v in dct.items() if k.startswith(prefix)}
