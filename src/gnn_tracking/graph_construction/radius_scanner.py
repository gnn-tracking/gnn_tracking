from __future__ import annotations

import math
import typing
from functools import cached_property

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from torch import Tensor
from torch_cluster import radius_graph
from torch_geometric.data import Data

from gnn_tracking.analysis.graphs import get_largest_segment_fracs
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.timing import Timer


def construct_graph(mo: dict[str, typing.Any], radius, max_num_neighbors=128):
    edge_index = radius_graph(mo["x"], radius, max_num_neighbors=max_num_neighbors)
    y: Tensor = (  # type: ignore
        mo["particle_id"][edge_index[0, :]] == mo["particle_id"][edge_index[1, :]]
    )
    data = Data(x=mo["x"], edge_index=edge_index, y=y)
    data.pt = mo["particle_id"]
    data.particle_id = mo["particle_id"]
    return data


class RSResults:
    def __init__(
        self,
        search_space: np.ndarray,
        results: np.ndarray,
        targets: typing.Sequence[float],
    ):
        self.search_space = search_space
        self.results = results
        self.targets = targets

    @cached_property
    def cs_r(self) -> CubicSpline:
        return CubicSpline(
            self.search_space[np.isfinite(self.results[:, 0])],
            self.results[np.isfinite(self.results[:, 0]), 0],
        )

    @cached_property
    def cs_n_edges(self) -> CubicSpline:
        return CubicSpline(self.search_space, self.results[:, 1])

    def get_target_radius(self, target: float) -> float:
        if target > max(self.results[:, 0]):
            return float("nan")
        return minimize(
            lambda radius: np.abs(self.cs_r(radius) - target),
            (self.search_space.min() + self.search_space.max()) / 2,
            bounds=((self.search_space.min(), self.search_space.max()),),
        ).x

    def get_foms(self) -> dict[str, float]:
        return {
            f"frac_segment50_{t*100:.0f}": self.cs_n_edges(
                self.get_target_radius(t)
            ).item()
            for t in self.targets
        }

    def plot(self) -> plt.Axes:
        xs = np.linspace(*self.search_space[[0, -1]], 1000)
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(xs, self.cs_r(xs), marker="none", color="C0", label="50frac")
        ax.plot(self.search_space, self.results[:, 0], "o", color="C0")
        ax2.plot(xs, self.cs_n_edges(xs), marker="none", color="C1", label="n_edges")
        ax2.plot(self.search_space, self.results[:, 1], "o", color="C1")
        for t in self.targets:
            ax.axhline(t, linestyle="--", lw=1, color="C0")
        for target in self.targets:
            ax.axvline(self.get_target_radius(target), linestyle="--", lw=1, color="C0")
        fig.legend()
        return ax


class RadiusScanner:
    def __init__(
        self,
        model_output: list[dict[str, typing.Any]],
        radius_range: tuple[float, float],
        max_num_neighbors: int = 128,
        n_trials: int = 10,
        max_edges: int = 5_000_000,
        target_fracs=(0.8, 0.9, 0.95),
        n_not_clipped=10,
    ):
        self._model_output = model_output
        self._radius_range = list(radius_range)
        self._max_num_neighbors = max_num_neighbors
        self._n_trials = n_trials
        self._study = None
        self.logger = get_logger("RadiusHP")
        self._max_edges = max_edges
        self._target_fracs = target_fracs
        self._n_not_clipped = n_not_clipped
        self._results: dict[float, tuple[float, int]] = {}

    def _objective_single(self, mo, radius: float) -> tuple[float, int | float]:
        if radius > self._radius_range[1]:
            return float("nan"), float("nan")
        data = construct_graph(
            mo, radius=radius, max_num_neighbors=self._max_num_neighbors
        )
        if data.num_edges > self._max_edges:
            self.logger.debug("Aborting trial for r=%f due to too many edges", radius)
            self._clip_radius_range(max_radius=radius)
            return float("nan"), data.num_edges
        r = (get_largest_segment_fracs(data) > 0.5).mean()
        return r, data.num_edges

    def _objective(self, radius) -> tuple[float, int | float]:
        """Objective function for optuna."""
        vals = []
        for mo in self._model_output:
            _r, _n_edges = self._objective_single(mo, radius)
            vals.append((_r, _n_edges))
        r, n_edges = np.array(vals).mean(axis=0)
        self.logger.debug("Radius %f -> 50-segment: %f, edges: %d", radius, r, n_edges)
        self._results[radius] = r, n_edges
        if r > max(self._target_fracs):
            self.logger.debug("Overachieving. Clipping")
            self._clip_radius_range(max_radius=radius)
        # Test if we are in the decreasing area
        smaller_radii = [r for r in self._results if r < radius]
        if len(smaller_radii) >= 1:
            nsr = max(smaller_radii)
            nsr_result = self._results[nsr][0]
            if r < nsr_result:
                self.logger.debug(
                    "Decreasing segment-connectivity for increased radius "
                    "(r=%f, r=%f). In random-sampling area. Clipping.",
                    nsr,
                    radius,
                )
                self._clip_radius_range(max_radius=radius)
        return r, n_edges

    def _clip_radius_range(self, min_radius=None, max_radius=None):
        if min_radius is not None:
            _ = self._radius_range[0]
            self._radius_range[0] = max(self._radius_range[0], min_radius)
            if not math.isclose(_, self._radius_range[0]):
                self.logger.debug("Updated min radius to %f", self._radius_range[0])
        if max_radius is not None:
            _ = self._radius_range[0]
            self._radius_range[1] = min(self._radius_range[1], max_radius)
            if not math.isclose(_, self._radius_range[1]):
                self.logger.debug("Updated max radius to %f", self._radius_range[1])
        assert self._radius_range[0] < self._radius_range[1]

    def _get_arrays(self):
        search_space = np.array(sorted(self._results))
        results = np.array([self._results[r] for r in search_space])
        return search_space, results

    def _update_search_range(self):
        search_space, results = self._get_arrays()
        min_radius = max(
            search_space[results[:, 0] < min(self._target_fracs)],
            default=self._radius_range[0],
        )
        max_radius = min(
            search_space[results[:, 0] > max(self._target_fracs)],
            default=self._radius_range[1],
        )
        self.logger.debug("Updating search range to %f, %f", min_radius, max_radius)
        self._clip_radius_range(
            min_radius=min_radius,
            max_radius=max_radius,
        )

    def __call__(self):
        initial_search_space = np.linspace(*self._radius_range, self._n_not_clipped)
        np.random.shuffle(initial_search_space)
        t = Timer()
        for radius in initial_search_space:
            self._objective(radius)
        elapsed = t()
        self.logger.info("Finished initial scan in %ds.", elapsed)
        self._update_search_range()
        search_space = np.linspace(*self._radius_range, self._n_trials)
        np.random.shuffle(search_space)
        self.logger.debug("Starting main scan.")
        for radius in search_space:
            self._objective(radius)
        elapsed = t()
        self.logger.info("Finished second scan in %ds.", elapsed)

        search_space, results = self._get_arrays()

        return RSResults(
            search_space=search_space,
            results=results,
            targets=self._target_fracs,
        )
