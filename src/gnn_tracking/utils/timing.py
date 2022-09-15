from __future__ import annotations

import logging
import timeit
from contextlib import contextmanager

from gnn_tracking.utils.log import get_logger


@contextmanager
def timing(name="Codeblock", logger=None):
    """Context manager for timing code blocks."""
    if logger is None:
        logger = get_logger("timing", level=logging.INFO)
    t0 = timeit.default_timer()
    try:
        yield
    finally:
        logger.info(f"{name} took {timeit.default_timer() - t0:.2f} seconds")
