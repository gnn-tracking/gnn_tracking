from __future__ import annotations

import sys

if sys.version_info >= (3, 9):
    # noinspection PyUnusedLocal
    from importlib import resources  # noqa: F401
else:
    # noinspection PyUnusedLocal
    # noinspection PyUnresolvedReferences
    import importlib_resources as resources  # noqa: F401
