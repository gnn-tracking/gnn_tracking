from __future__ import annotations

import logging

from gnn_tracking.utils.log import get_logger, logger


def test_logger():
    logger.info("Hello world")


def test_get_logger():
    assert isinstance(get_logger(), logging.Logger)
