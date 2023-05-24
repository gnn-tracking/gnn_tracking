from __future__ import annotations

from functools import partial
from typing import Sequence

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
        r_averaged.append(
            {
                k: sum([x[k] for x in r[i :: len(thresholds)]]) / n_batches
                for k in r[0].keys()
            }
        )
    return pd.DataFrame.from_records(r_averaged)


def plot_threshold_vs_track_info(df: pd.DataFrame) -> plt.Axes:
    """Plots the output of `collect_all_ec_stats`."""
    fig, ax = plt.subplots()
    markup = dict(marker=".")
    ax.plot(df.threshold, df.frac_perfect, **markup, label="Single segment", c="C0")
    ax.plot(df.threshold, df.frac_segment50, **markup, label="50% segment", c="C2")
    ax.plot(
        df.threshold,
        df.frac_component50,
        **markup,
        label="50% component",
        ls="--",
        c="C2",
        markerfacecolor="none",
    )
    ax.plot(df.threshold, df.frac_segment75, **markup, label="75% segment", c="C1")
    ax.plot(
        df.threshold,
        df.frac_component75,
        **markup,
        label="75% component",
        ls="--",
        c="C1",
        markerfacecolor="none",
    )
    ax.plot(df.threshold, df.TPR_thld, c="C3", label="TPR ($p_T > 0.9$ GeV)", **markup)
    ax.plot(df.threshold, df.FPR, c="C3", label="FPR", ls="--", **markup)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("EC threshold")
    ax.axhline(0.9, c="gray", alpha=0.3, lw=1)
    ax.axhline(0.95, c="gray", alpha=0.3, lw=1)
    ax.axhline(0.85, c="gray", alpha=0.3, lw=1)

    ax.legend()
    return ax
