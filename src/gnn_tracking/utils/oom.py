import collections
import functools
from typing import Callable

import torch.cuda

from gnn_tracking.utils.log import logger

N_OOM_ERRORS = collections.defaultdict(int)


def is_oom_error(e: Exception) -> bool:
    """Is this an out of memory (OOM) error?"""
    if isinstance(e, RuntimeError) and "out of memory" in str(e):
        return True
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return False


def tolerate_some_oom_errors(fct: Callable):
    """Decorators to tolerate a couple of out of memory (OOM) errors."""
    max_errors = 10

    @functools.wraps(fct)
    def wrapped_fct(*args, **kwargs):
        try:
            result = fct(*args, **kwargs)
        except RuntimeError as e:
            if is_oom_error(e):
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
