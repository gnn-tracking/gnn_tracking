from __future__ import annotations

import sys

if sys.version_info >= (3, 9):
    # noinspection PyUnusedLocal
    pass
else:
    # noinspection PyUnusedLocal
    # noinspection PyUnresolvedReferences
    import importlib_resources as resources  # noqa: F401
