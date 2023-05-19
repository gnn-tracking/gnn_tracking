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


def get_tpr_fpr(
    threshold: float,
    w: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    passes = w >= threshold
    true = y == 1
    tp = (passes & true).sum()
    fp = (passes & (~true)).sum()
    tpr = tp.sum().item() / sum(true)
    fpr = fp.sum().item() / sum(~true)
    return {"tpr": tpr.item(), "fpr": fpr.item()}


def get_all_ec_stats(threshold: float, w: torch.Tensor, data: Data) -> dict[str, float]:
    """See `collect_ec_stats`"""
    return (
        {"threshold": threshold}
        | get_tpr_fpr(threshold, w, data.y)
        | summarize_track_graph_info(
            get_track_graph_info_from_data(data, w, threshold=threshold)
        )
    )


def collect_all_ec_stats(
    model: torch.nn.Module,
    data_loader: DataLoader,
    thresholds: Sequence[float],
    n_batches: int | None = None,
) -> pd.DataFrame:
    model.eval()
    r = []
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            w = model(data)["W"]
            r += process_map(
                partial(get_all_ec_stats, w=w, data=data), thresholds, max_workers=6
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
    """Plots the output of `collect_ec_stats`."""
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
    ax.plot(df.threshold, df.tpr, c="C3", label="TPR", **markup)
    ax.plot(df.threshold, df.fpr, c="C3", label="FPR", ls="--", **markup)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("EC threshold")
    ax.axhline(0.9, c="gray", alpha=0.3, lw=1)
    ax.axhline(0.95, c="gray", alpha=0.3, lw=1)
    ax.axhline(0.85, c="gray", alpha=0.3, lw=1)

    ax.legend()
    return ax
