"""Plotting functions to plot the latent space"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gnn_tracking.utils.colors import lighten_color


def plot_coordinates_flat(x: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot all hits in the latent space in two dimensions.
    If ``x`` has three dimensions, the third dimension is plotted as color.

    Args:
        x: Any coordinates of dimension > 3 are ignored
        ax:

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    c = None
    if x.shape[1] == 3:
        c = x[:, 2]
    cax = ax.scatter(x[:, 0], x[:, 1], c=c, s=2)
    fig.colorbar(cax)
    if x.shape[1] == 3:
        ax.set_title("Color is third dimension")
    return ax


def plot_coordinates_3d(
    x: np.ndarray, pid: np.ndarray, ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot all hits in the latent space in three dimensions.
    If the latent space dimension is larger than three, reduce the dimensionality
    of x.

    The color of the points is the particle ID, noise is plotted red.

    Args:
        x: Any coordinates of dimension > 3 are ignored
        pid:
        ax
    """
    if not x.shape[1] >= 3:
        raise ValueError(
            "This plot function only works for latent space dimension >= 3"
        )
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    noise_mask = pid <= 0
    nnx = x[~noise_mask]
    ax.scatter3D(nnx[:, 0], nnx[:, 1], nnx[:, 2], s=2, c=pid[~noise_mask])
    nx = x[noise_mask]
    ax.scatter3D(nx[:, 0], nx[:, 1], nx[:, 2], s=2, c="red")
    return ax


def _draw_circles(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, colors, eps=1) -> None:
    assert xs.shape == ys.shape
    for x, y, c in zip(xs, ys, colors):
        circle = plt.Circle(
            (x, y), eps, facecolor=lighten_color(c, 0.2), linestyle="none"
        )
        ax.add_patch(circle)


def plot_selected_pids(
    x: np.ndarray, pid: np.ndarray, ax: plt.Axes | None = None
) -> plt.Axes:
    """Draw latent space, highlighting 10 randomly selected PIDs. In addition, shaded
    circles of radius 1 are drawn around each hit for the selected PIDs.

    Args:
        x: Coordinates in latent space. Only the first two coordinates are considered.
        pid:
        ax:

    Returns:

    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_mapper = np.vectorize(colors.__getitem__)
    # todo: mark condensation point
    selected_pids = np.random.choice(pid[pid > 0], 10).astype("int64")
    # map PIDs to number 0 to #PIDs - 1
    pid_mapper = np.vectorize({p.item(): i for i, p in enumerate(selected_pids)}.get)
    mask = np.isin(pid, selected_pids)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    c = color_mapper(pid_mapper(pid[mask]))
    _draw_circles(ax, x[mask][:, 0], x[mask][:, 1], c)
    ax.scatter(
        x[~mask][:, 0], x[~mask][:, 1], c="silver", alpha=1, label="Other hits", s=2
    )
    ax.scatter(x[mask][:, 0], x[mask][:, 1], c=c, label="Hits of selected PIDs", s=2)
    fig.legend()
    return ax
