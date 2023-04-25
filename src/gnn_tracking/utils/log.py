# std
from __future__ import annotations

import logging

import colorlog

LOG_DEFAULT_LEVEL = logging.DEBUG


def get_logger(name="gnn-tracking", level=LOG_DEFAULT_LEVEL):
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
    #  This is not the same as just setting name="" in the fct arguments.
    #  This would set the root logger to debug mode, which for example causes
    #  the matplotlib font manager (which uses the root logger) to throw lots of
    #  messages. Here, we want to keep our named logger, but just drop the
    #  name.
    name_incl = "" if name == "gnn-tracking" else f" {name}"
    formatter = colorlog.ColoredFormatter(
        f"%(log_color)s[%(asctime)s{name_incl}] %(levelname)s: %(message)s",
        log_colors=log_colors,
        datefmt="%H:%M:%S",
    )
    sh.setFormatter(formatter)
    # Controlled by overall logger level
    sh.setLevel(logging.DEBUG)

    _log.addHandler(sh)

    return _log


logger = get_logger()
