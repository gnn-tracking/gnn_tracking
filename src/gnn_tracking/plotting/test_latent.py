from __future__ import annotations

import dataclasses

import numpy as np

from gnn_tracking.plotting.latent import plot_coordinates_3d, plot_coordinates_flat


@dataclasses.dataclass
class _TestData:
    x: np.ndarray
    pid: np.ndarray


_test_data = _TestData(x=np.random.rand(100, 3), pid=np.random.randint(0, 10, 100))


def test_draw_coordinates_flat():
    plot_coordinates_flat(_test_data.x)


def test_draw_coordinates_3d():
    plot_coordinates_3d(_test_data.x, _test_data.pid)


# def test_draw_selected_pids():
#     spp = SelectedPidsPlot(_test_data.x, _test_data.pid, data=data, labels=labels,
#                            ec_hit_mask=model_output["ec_hit_mask"],
#                            selected_pids=random_pids)
#     fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
#     spp.plot_other_hit_latent(axs[0])
#     spp.plot_selected_pid_latent(axs[0])
#     spp.plot_collateral_latent(axs[0])
#     spp.plot_other_hit_ep(axs[1])
#     spp.plot_selected_pid_ep(axs[1])
#     spp.plot_collateral_ep(axs[1])
