import collections
import functools
from typing import Callable

from gnn_tracking.utils.log import logger

N_OOM_ERRORS = collections.defaultdict(int)


def tolerate_some_oom_errors(fct: Callable):
    """Decorators to tolerate a couple of out of memory (OOM) errors."""
    max_errors = 10

    @functools.wraps(fct)
    def wrapped_fct(*args, **kwargs):
        try:
            result = fct(*args, **kwargs)
        except RuntimeError as e:
            if e == RuntimeError and "out of memory" in str(e):
                logger.warning(
                    "WARNING: ran out of memory (OOM), skipping batch. "
                    "If this happens frequently, decrease the batch size. "
                    f"Will abort if we get {max_errors} consecutive OOM errors."
                )
                N_OOM_ERRORS[fct.__name__] += 1
                if N_OOM_ERRORS[fct.__name__] > max_errors:
                    raise
                return None
            raise
        else:
            N_OOM_ERRORS[fct.__name__] = 0
            return result

    return wrapped_fct
