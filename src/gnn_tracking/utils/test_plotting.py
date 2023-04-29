from __future__ import annotations

from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from trackml.dataset import load_event

from gnn_tracking.test_data import trackml_test_data_dir, trackml_test_data_prefix
from gnn_tracking.utils.plotting import EventPlotter


@patch("matplotlib.pyplot.show")
def test_event_plotter(mock_show):
    assert mock_show is plt.show
    path = trackml_test_data_dir
    event = trackml_test_data_prefix
    evtid = 1

    def calc_eta(r, z):
        return -1.0 * np.log(np.tan(np.arctan2(r, z) / 2.0))

    hits, particles, truth = load_event(
        path / event, parts=["hits", "particles", "truth"]
    )
    particles["pt"] = np.sqrt(particles.px**2 + particles.py**2)
    particles["eta_pt"] = calc_eta(particles.pt, particles.pz)
    truth = truth[["hit_id", "particle_id"]].merge(
        particles[["particle_id", "pt", "eta_pt", "q", "vx", "vy"]],
        on="particle_id",
    )
    hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
    hits["phi"] = np.arctan2(hits.y, hits.x)
    hits["eta"] = calc_eta(hits.r, hits.z)
    hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
    hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)
    hits = hits[
        ["hit_id", "r", "phi", "eta", "x", "y", "z", "u", "v", "volume_id"]
    ].merge(truth[["hit_id", "particle_id", "pt", "eta_pt"]], on="hit_id")

    plotter = EventPlotter(indir=path)
    fig, _ = plotter.plot_ep_rv_uv(evtid=evtid)
    plotted_data_points = []
    original_data_points = []
    original_data_points.append(hits[["eta", "phi"]].values)
    original_data_points.append(hits[["z", "r"]].values)
    original_data_points.append(hits[["u", "v"]].values)
    for i in range(3):
        plotted_data_points.append(np.array(fig.axes[i].get_lines()[0].get_xydata()))
    plt.close(fig)
    assert (plotted_data_points[0] == original_data_points[0]).all()
    assert (plotted_data_points[1] == original_data_points[1]).all()
    assert (plotted_data_points[2] == original_data_points[2]).all()
