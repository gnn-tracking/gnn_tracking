from __future__ import annotations

from functools import partial
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data, DataLoader
from tqdm.contrib.concurrent import process_map

from gnn_tracking.analysis.graphs import (
    get_track_graph_info_from_data,
    summarize_track_graph_info,
)
from gnn_tracking.metrics.binary_classification import BinaryClassificationStats
from gnn_tracking.utils.dictionaries import add_key_suffix
from gnn_tracking.utils.graph_masks import get_edge_mask_from_node_mask


def get_all_ec_stats(
    threshold: float, w: torch.Tensor, data: Data, pt_thld=0.9
) -> dict[str, float]:
    """Evaluate edge classification performance for a single threshold and a single
    batch.

    Args:
        threshold: Edge classification threshold
        w: Edge classification output
        data: Data
        pt_thld: pt threshold for particle IDs to consider. For edge classification
            stats (TPR, etc.), two versions are calculated: The stats ending with
            `_thld` are calculated for all edges with pt > pt_thld

    Returns:
        Dictionary of metrics
    """
    pt_edge_mask = get_edge_mask_from_node_mask(data.pt > pt_thld, data.edge_index)
    bcs_thld = BinaryClassificationStats(
        output=w[pt_edge_mask], y=data.y[pt_edge_mask].long(), thld=threshold
    )
    bcs = BinaryClassificationStats(output=w, y=data.y.long(), thld=threshold)
    return (
        {"threshold": threshold}
        | bcs.get_all()
        | add_key_suffix(bcs_thld.get_all(), "_thld")
        | summarize_track_graph_info(
            get_track_graph_info_from_data(data, w, threshold=threshold)
        )
    )


def collect_all_ec_stats(
    model: torch.nn.Module,
    data_loader: DataLoader,
    thresholds: Sequence[float],
    n_batches: int | None = None,
    max_workers=6,
) -> pd.DataFrame:
    """Collect edge classification statistics for a model and a data loader, basically
    mapping `get_all_ec_stats` over the data loader with multiprocessing.

    Args:
        model: Edge classifier model
        data_loader: Data loader
        thresholds: List of EC thresholds to evaluate
        n_batches: Number of batches to evaluate
        max_workers: Number of workers for multiprocessing

    Returns:
        DataFrame with columns as in `get_all_ec_stats`
    """
    model.eval()
    r = []
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            w = model(data)["W"]
            r += process_map(
                partial(get_all_ec_stats, w=w, data=data),
                thresholds,
                max_workers=max_workers,
            )
            if n_batches is not None and idx >= n_batches - 1:
                break

    r_averaged = []
    n_batches = len(r) // len(thresholds)
    for i in range(len(thresholds)):
        batched = {
            k: np.array([x[k] for x in r[i :: len(thresholds)]]) for k in r[0].keys()
        }
        r_averaged.append(
            {k: np.mean(batch) for k, batch in batched.items()}
            | {
                f"{k}_err": np.std(batch) / np.sqrt(n_batches)
                for k, batch in batched.items()
            }
        )
    return pd.DataFrame.from_records(r_averaged)


class ThresholdTrackInfoPlot:
    def __init__(self, df: pd.DataFrame):
        """Plot track info as a function of EC threshold.

        To get the plot in one go, simply call the `plot` method. Alternatively,
        use the individual methods to plot the different components separately.

        Args:
            df: DataFrame with columns as in `get_all_ec_stats`
        """
        self.df = df
        self.ax: plt.Axes | None = None

    def plot(self):
        """Plot all the things."""
        self.setup_axes()
        self.plot_single_segment()
        self.plot_50()
        self.plot_75()
        self.plot_tpr_fpr()
        self.plot_hlines()
        self.plot_mcc()
        self.add_legend()

    def setup_axes(self):
        _, ax = plt.subplots()
        ax.set_xlabel("EC threshold")
        self.ax = ax

    def plot_single_segment(self):
        self.ax.errorbar(
            self.df.threshold,
            self.df.frac_perfect,
            yerr=self.df.frac_perfect_err,
            label="Single segment",
        )

    def plot_50(self):
        line, *_ = self.ax.errorbar(
            self.df.threshold,
            self.df.frac_segment50,
            yerr=self.df.frac_segment50_err,
            label="50% segment",
        )
        self.ax.plot(
            self.df.threshold,
            self.df.frac_component50,
            label="50% component",
            ls="--",
            c=line.get_color(),
            markerfacecolor="none",
        )

    def plot_75(self):
        line, *_ = self.ax.errorbar(
            self.df.threshold,
            self.df.frac_segment75,
            yerr=self.df.frac_segment75_err,
            label="75% segment",
        )
        self.ax.plot(
            self.df.threshold,
            self.df.frac_component75,
            label="75% component",
            ls="--",
            c=line.get_color(),
            markerfacecolor="none",
        )

    def plot_tpr_fpr(self):
        line, *_ = self.ax.errorbar(
            self.df.threshold,
            self.df.TPR_thld,
            yerr=self.df.TPR_thld_err,
            label="TPR ($p_T > 0.9$ GeV)",
        )
        self.ax.errorbar(
            self.df.threshold,
            self.df.FPR,
            yerr=self.df.FPR_err,
            c=line.get_color(),
            label="FPR",
            ls="--",
        )

    def plot_mcc(self):
        self.ax.errorbar(
            self.df.threshold,
            self.df.MCC_thld,
            self.df.MCC_thld_err,
            label="MCC ($p_T > 0.9$ GeV)",
        )

    def plot_hlines(self):
        self.ax.axhline(0.9, c="gray", alpha=0.3, lw=1)
        self.ax.axhline(0.95, c="gray", alpha=0.3, lw=1)
        self.ax.axhline(0.85, c="gray", alpha=0.3, lw=1)

    def add_legend(self):
        self.ax.legend()
