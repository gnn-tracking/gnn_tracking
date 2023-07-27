import logging
import timeit
from contextlib import contextmanager

from gnn_tracking.utils.log import get_logger


class Timer:
    def __init__(self):
        """Helper class for timing code blocks.
        ``t0`` is set to the current time when the class is instantiated.
        """
        self.t0 = timeit.default_timer()

    def __call__(self):
        """Return the time since the last call or ``__init__``."""
        now = timeit.default_timer()
        timedelta = now - self.t0
        self.t0 = now
        return timedelta


@contextmanager
def timing(name="Codeblock", logger=None):
    """Context manager for timing code blocks."""
    if logger is None:
        logger = get_logger("timing", level=logging.INFO)
    t = Timer()
    try:
        yield
    finally:
        logger.info("%s took %f seconds", name, t())
