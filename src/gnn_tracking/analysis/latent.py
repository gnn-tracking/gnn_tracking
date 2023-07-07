"""Plotting functions to plot the latent space"""


from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor as T

from gnn_tracking.utils.colors import lighten_color
from gnn_tracking.utils.log import logger


def get_color_mapper(
    selected_values: Sequence, colors: Sequence | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Get a function that maps values to colors."""
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(selected_values) > len(colors):
        _ = (
            f"Only up to {len(colors)} values can be selected because we only have "
            f"that many colors configured."
        )
        raise ValueError(_)

    # cm = np.vectorize(colors.__getitem__)
    # vm = np.vectorize({p.item(): i for i, p in enumerate(selected_values)}.get)

    color_map = {p.item(): colors[i] for i, p in enumerate(selected_values)}

    def mapper(values):
        return np.array([color_map.get(v.item()) for v in values])

    return mapper


class SelectedPidsPlot:
    def __init__(
        self,
        *,
        condensation_space: T,
        particle_id: T,
        labels: T,
        selected_pids: Sequence[int] | None = None,
        ec_hit_mask: T,
        input_node_features: T,
    ):
        """Plot the condensation space with selected PIDs highlighted.
        Two kinds of plots are supported: Latent space coordinates and phi/eta.
        For each of these, separate methods plot hits of the selected PIDs,
        all other hits, and collateral hits (hits in the same cluster as the
        selected PIDs).

        Args:
            condensation_space:
            particle_id:
            labels:
            selected_pids:
            ec_hit_mask: If we do orphan node prediction, we need to know which hits
                make it to the condensation space
            input_node_features
        """
        if ec_hit_mask is None:
            ec_hit_mask = torch.ones_like(particle_id).bool()
        self._ec_hit_mask = ec_hit_mask
        self._x = condensation_space
        self._pids = particle_id[self._ec_hit_mask]
        self._labels = labels
        if selected_pids is None:
            logger.warning(
                "No PIDs selected, using random PIDs (no pt threshold applied). "
            )
            selected_pids = torch.Tensor(
                np.random.choice(self._pids[self._pids > 0], 10).astype("int64")
            )
        self._selected_pids = selected_pids

        self._color_mapper = get_color_mapper(selected_pids)
        self._selected_pid_mask = torch.isin(self._pids, self._selected_pids)

        self._phi = input_node_features[self._ec_hit_mask, 3]
        self._eta = input_node_features[self._ec_hit_mask, 1]

    def get_collateral_mask(self, pid: int) -> T:
        """Mask for hits that are in the same cluster(s) as the hits belonging to this
        particle ID.
        """
        assert self._labels is not None
        pid_mask = self._pids == pid
        assoc_labels = torch.unique(self._labels[pid_mask])
        label_mask = torch.isin(self._labels, assoc_labels)
        return label_mask & (~pid_mask)

    @staticmethod
    def plot_circles(ax: plt.Axes, xs: T, ys: T, colors, eps=1) -> None:
        assert xs.shape == ys.shape
        for x, y, c in zip(xs, ys, colors):
            circle = plt.Circle(
                (x, y), eps, facecolor=lighten_color(c, 0.2), linestyle="none"
            )
            ax.add_patch(circle)

    def get_colors(self, pids: T | Sequence) -> Sequence:
        return self._color_mapper(pids)

    def plot_selected_pid_latent(self, ax: plt.Axes, plot_circles=False) -> None:
        # todo: mark condensation point
        mask = self._selected_pid_mask
        c = self.get_colors(self._pids[mask])
        if plot_circles:
            self.plot_circles(ax, self._x[mask][:, 0], self._x[mask][:, 1], c)
        ax.scatter(
            self._x[mask][:, 0],
            self._x[mask][:, 1],
            c=c,
            label="Hits of selected PIDs",
            s=12,
        )

    def plot_collateral_latent(self, ax: plt.Axes) -> None:
        for pid in self._selected_pids:
            mask = self.get_collateral_mask(pid)
            ax.scatter(
                self._x[mask][:, 0],
                self._x[mask][:, 1],
                c=self.get_colors([pid]),
                alpha=1,
                label="Collateral",
                s=12,
                marker="x",
            )

    def plot_other_hit_latent(self, ax: plt.Axes) -> None:
        mask = self._selected_pid_mask
        ax.scatter(
            self._x[~mask][:, 0],
            self._x[~mask][:, 1],
            c="silver",
            alpha=1,
            label="Other hits",
            s=2,
        )

    def plot_selected_pid_ep(self, ax: plt.Axes) -> None:
        mask = self._selected_pid_mask
        ax.scatter(
            self._phi[mask],
            self._eta[mask],
            c=self.get_colors(self._pids[mask]),
            s=12,
            label="Selected PIDs",
        )

    def plot_other_hit_ep(self, ax: plt.Axes) -> None:
        mask = ~self._selected_pid_mask
        ax.scatter(
            self._phi[mask],
            self._eta[mask],
            c="silver",
            s=2,
            label="Other hits",
        )

    def plot_collateral_ep(self, ax: plt.Axes) -> None:
        for pid in self._selected_pids:
            mask = self.get_collateral_mask(pid)
            ax.scatter(
                self._phi[mask],
                self._eta[mask],
                c=self.get_colors([pid]),
                alpha=1,
                s=12,
                marker="x",
            )
