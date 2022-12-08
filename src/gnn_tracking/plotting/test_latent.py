from __future__ import annotations

import dataclasses

import numpy as np

from gnn_tracking.plotting.latent import (
    plot_coordinates_3d,
    plot_coordinates_flat,
    plot_selected_pids,
)


@dataclasses.dataclass
class _TestData:
    x: np.ndarray
    pid: np.ndarray


_test_data = _TestData(x=np.random.rand(100, 3), pid=np.random.randint(0, 10, 100))


def test_draw_coordinates_flat():
    plot_coordinates_flat(_test_data.x)


def test_draw_coordinates_3d():
    plot_coordinates_3d(_test_data.x, _test_data.pid)


def test_draw_selected_pids():
    plot_selected_pids(_test_data.x, _test_data.pid)
