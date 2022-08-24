from __future__ import annotations

from gnn_tracking.test_data import test_data_dir


def test_test_data_dir():
    assert test_data_dir.is_dir()
