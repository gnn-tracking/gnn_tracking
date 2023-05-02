from __future__ import annotations

from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from numpy.testing import assert_approx_equal

from gnn_tracking.test_data import trackml_test_data_dir
from gnn_tracking.utils.plotting import EventPlotter


@patch("matplotlib.pyplot.show")
def test_event_plotter(mock_show):
    assert mock_show is plt.show
    path = trackml_test_data_dir
    evtid = 1

    plotter = EventPlotter(indir=path)
    fig, _ = plotter.plot_ep_rv_uv(evtid=evtid)
    original_data_points = [
        [
            [-3.65964174, -2.99420524],
            [-3.14542103, -2.94588566],
            [-3.65947032, -2.99451041],
            [-3.14519644, -2.94551826],
            [-2.90377235, -2.63840151],
            [-3.21849656, -2.51005006],
            [-3.2187562, -2.50965071],
            [-3.48590827, -2.21800613],
            [-2.85355783, -2.35087061],
            [-3.21562123, -2.23991871],
        ],
        [
            [-1502.5, 77.40522003],
            [-1498.0, 129.21363831],
            [-1498.0, 77.18669128],
            [-1502.0, 129.58804321],
            [-1498.0, 164.72381592],
            [-1502.0, 120.39829254],
            [-1498.0, 120.04631805],
            [-1498.0, 91.84147644],
            [-1502.0, 173.72445679],
            [-1498.0, 120.42472839],
        ],
        [
            [-0.01277896, -0.00189722],
            [-0.00759138, -0.00150495],
            [-0.01281572, -0.00189868],
            [-0.0075689, -0.00150338],
            [-0.00531829, -0.00292747],
            [-0.00670373, -0.00490364],
            [-0.00672142, -0.00492071],
            [-0.00656526, -0.00868637],
            [-0.00404855, -0.00409189],
            [-0.00515092, -0.00651333],
        ],
    ]
    for i in range(3):
        assert_approx_equal(
            np.array(fig.axes[i].get_lines()[0].get_xydata()[:10]),
            np.array(original_data_points[i]),
        )
    plt.close(fig)
