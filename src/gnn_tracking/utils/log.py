# std
from __future__ import annotations

import logging

import colorlog

LOG_DEFAULT_LEVEL = logging.DEBUG

logging.captureWarnings(True)


def get_logger(name="GNNTracking", level=LOG_DEFAULT_LEVEL):
    """Sets up global logger."""
    _log = colorlog.getLogger(name)

    if _log.handlers:
        # the logger already has handlers attached to it, even though
        # we didn't add it ==> logging.get_logger got us an existing
        # logger ==> we don't need to do anything
        return _log

    _log.setLevel(level)

    sh = colorlog.StreamHandler()
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s", log_colors=log_colors
    )
    sh.setFormatter(formatter)
    # Controlled by overall logger level
    sh.setLevel(logging.DEBUG)

    _log.addHandler(sh)

    return _log


logger = get_logger()
