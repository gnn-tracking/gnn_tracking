import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from gnn_tracking.analysis.latent import SelectedPidsPlot


@pytest.fixture()
def selected_pids_test_data() -> dict[str, np.ndarray]:
    kwargs = {}
    for key in ["condensation_space", "input_node_features"]:
        kwargs[key] = torch.rand(size=(100, 5))
    for key in ["particle_id", "labels"]:
        kwargs[key] = torch.randint(10, size=(100,))
    kwargs["ec_hit_mask"] = torch.ones(100).bool()
    return kwargs


def test_draw_selected_pids(selected_pids_test_data):
    spp = SelectedPidsPlot(**selected_pids_test_data)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    spp.plot_other_hit_latent(axs[0])
    spp.plot_selected_pid_latent(axs[0])
    spp.plot_collateral_latent(axs[0])
    spp.plot_other_hit_ep(axs[1])
    spp.plot_selected_pid_ep(axs[1])
    spp.plot_collateral_ep(axs[1])
