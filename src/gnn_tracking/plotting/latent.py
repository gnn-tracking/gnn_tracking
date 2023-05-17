"""Plotting functions to plot the latent space"""

from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from gnn_tracking.utils.colors import lighten_color
from gnn_tracking.utils.log import logger


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
    x: np.ndarray,
    pid: np.ndarray,
    selected_pids: Sequence[int] | None = None,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw latent space, highlighting 10 randomly selected PIDs. In addition, shaded
    circles of radius 1 are drawn around each hit for the selected PIDs.

    Args:
        x: Coordinates in latent space. Only the first two coordinates are considered.
        pid:
        selected_pids: PIDs to highlight. If None, random PIDs are used
        ax:

    Returns:

    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_mapper = np.vectorize(colors.__getitem__)
    # todo: mark condensation point
    if selected_pids is None:
        selected_pids = np.random.choice(pid[pid > 0], 10).astype("int64")
    else:
        if len(selected_pids) > 10:
            raise ValueError("Only up to 10 PIDs can be specified.")
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


def get_color_mapper(
    selected_values: Sequence, colors: Sequence | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Get a function that maps values to colors."""
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(selected_values) > len(colors):
        raise ValueError(f"Only up to {len(colors)} values can be selected.")

    # cm = np.vectorize(colors.__getitem__)
    # vm = np.vectorize({p.item(): i for i, p in enumerate(selected_values)}.get)

    color_map = {p.item(): colors[i] for i, p in enumerate(selected_values)}

    def mapper(values):
        return np.array([color_map.get(v.item()) for v in values])

    return mapper


class SelectedPidsPlot:
    def __init__(
        self,
        x_latent: np.ndarray,
        pid: np.ndarray,
        labels: np.ndarray | None = None,
        selected_pids: Sequence[int] | None = None,
        data: np.ndarray | None = None,
        ec_hit_mask: np.ndarray | None = None,
    ):
        self.data = data
        if ec_hit_mask is None:
            ec_hit_mask = np.ones_like(pid, dtype=bool)
        self._ec_hit_mask = ec_hit_mask
        self.x = x_latent
        self.pid = pid[self._ec_hit_mask]
        self.labels = labels
        if selected_pids is None:
            logger.warning(
                "No PIDs selected, using random PIDs (no pt threshold applied). "
            )
            selected_pids = np.random.choice(self.pid[self.pid > 0], 10).astype("int64")
        self.selected_pids = selected_pids

        self._color_mapper = get_color_mapper(selected_pids)
        self._selected_pid_mask = np.isin(self.pid, self.selected_pids)

        self._phi = self.data.x[self._ec_hit_mask, 3]
        self._eta = self.data.x[self._ec_hit_mask, 1]

    def get_collateral_mask(self, pid: int) -> np.ndarray:
        """Mask for hits that are in the same cluster(s) as the hits belonging to this
        particle ID.
        """
        assert self.labels is not None
        pid_mask = (self.pid == pid).numpy()
        assoc_labels = np.unique(self.labels[pid_mask])
        label_mask = np.isin(self.labels, assoc_labels)
        col_mask = label_mask & (~pid_mask)
        return col_mask

    @staticmethod
    def plot_circles(
        ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, colors, eps=1
    ) -> None:
        assert xs.shape == ys.shape
        for x, y, c in zip(xs, ys, colors):
            circle = plt.Circle(
                (x, y), eps, facecolor=lighten_color(c, 0.2), linestyle="none"
            )
            ax.add_patch(circle)

    def get_colors(self, pids: np.ndarray | Sequence) -> np.ndarray:
        return self._color_mapper(pids)

    def plot_selected_pid_latent(self, ax: plt.Axes, plot_circles=False) -> None:
        # todo: mark condensation point
        mask = self._selected_pid_mask
        c = self.get_colors(self.pid[mask])
        if plot_circles:
            self.plot_circles(ax, self.x[mask][:, 0], self.x[mask][:, 1], c)
        ax.scatter(
            self.x[mask][:, 0],
            self.x[mask][:, 1],
            c=c,
            label="Hits of selected PIDs",
            s=12,
        )

    def plot_collateral_latent(self, ax):
        for pid in self.selected_pids:
            mask = self.get_collateral_mask(pid)
            ax.scatter(
                self.x[mask][:, 0],
                self.x[mask][:, 1],
                c=self.get_colors([pid]),
                alpha=1,
                label="Collateral",
                s=12,
                marker="x",
            )

    def plot_other_hit_latent(self, ax):
        mask = self._selected_pid_mask
        ax.scatter(
            self.x[~mask][:, 0],
            self.x[~mask][:, 1],
            c="silver",
            alpha=1,
            label="Other hits",
            s=2,
        )

    def plot_selected_pid_ep(self, ax):
        mask = self._selected_pid_mask
        ax.scatter(
            self._phi[mask],
            self._eta[mask],
            c=self.get_colors(self.pid[mask]),
            s=12,
            label="Selected PIDs",
        )

    def plot_other_hit_ep(self, ax):
        mask = ~self._selected_pid_mask
        ax.scatter(
            self._phi[mask],
            self._eta[mask],
            c="silver",
            s=2,
            label="Other hits",
        )

    def plot_collateral_ep(self, ax):
        for pid in self.selected_pids:
            mask = self.get_collateral_mask(pid)
            ax.scatter(
                self._phi[mask],
                self._eta[mask],
                c=self.get_colors([pid]),
                alpha=1,
                s=12,
                marker="x",
            )
