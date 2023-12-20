import numpy as np
import pandas as pd

from gnn_tracking.analysis.efficiencies import (
    PerformanceComparisonPlot,
    PerformancePlot,
    TracksVsDBSCANPlot,
)


def test_track_vs_dbscan_parameters():
    mean_df = pd.DataFrame(
        {
            "eps": [0.1],
            "min_samples": [1],
            "test": [1],
            "test_std": [0.5],
        }
    )
    p = TracksVsDBSCANPlot(
        mean_df,
        watermark="test",
        model="test",
    )
    p.plot_var("test")


def test_performance_plot():
    df = pd.DataFrame(
        {
            "test": [1, 0.5, 0.25],
            "test_err": [0.5, 0.25, 0.125],
        }
    )
    pt = [1, 2, 3, 4]
    p = PerformancePlot(xs=np.array(pt), df=df, df_ul=df)
    p.add_blocked(1, 2)
    p.plot_var("test", color="red")
    p.add_legend()


def test_performance_comparison_plot():
    df = pd.DataFrame(
        {
            "test": [1, 0.5, 0.25],
            "test_err": [0.5, 0.25, 0.125],
        }
    )
    pt = [1, 2, 3, 4]
    p = PerformanceComparisonPlot(var="test", x_label="test", xs=np.array(pt))
    p.add_blocked(1, 2)
    p.plot_var(df, color="red", label="test")
    p.add_legend()
