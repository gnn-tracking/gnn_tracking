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
        mo["particle_id"][edge_index[0]] == mo["particle_id"][edge_index[1]]
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

    def _get_fom(self, target: float) -> tuple[float, float]:
        if np.isfinite(self.search_space).sum() < 2:
            return float("nan"), float("nan")
        target_r = self.get_target_radius(target)
        if math.isnan(target_r):
            return float("nan"), float("nan")
        return target_r, self.cs_n_edges(target_r).item()

    def get_foms(self) -> dict[str, float]:
        foms = {}
        for t in self.targets:
            r, n_edges = self._get_fom(t)
            foms[f"n_edges_frac_segment50_{t*100:.0f}"] = n_edges
            foms[f"n_edges_frac_segment50_{t*100:.0f}_r"] = r
        idx_max_frac = self.results[:, 0].argmax()
        foms["max_frac_segment50"] = self.results[idx_max_frac, 0]
        foms["n_edges_max_frac_segment50"] = self.results[idx_max_frac, 1]
        foms["max_frac_segment50_r"] = self.search_space[idx_max_frac]
        return foms

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
        start_radii: list[float] | None = None,
    ):
        if start_radii is None:
            start_radii = []
        self._start_radii = start_radii
        self._model_output = model_output
        self._radius_range = list(radius_range)
        self._max_num_neighbors = max_num_neighbors
        self._n_trials = n_trials
        self._study = None
        self.logger = get_logger("RadiusHP")
        self._max_edges = max_edges
        self._targets = target_fracs
        self._results: dict[float, tuple[float, int]] = {}

    def _objective_single(
        self, mo: dict[str, typing.Any], radius: float
    ) -> tuple[float, int | float]:
        if radius > self._radius_range[1]:
            return float("nan"), float("nan")
        data = construct_graph(
            mo, radius=radius, max_num_neighbors=self._max_num_neighbors
        )
        if data.num_edges > self._max_edges:
            self._clip_radius_range(max_radius=radius, reason="too many edges")
            return float("nan"), data.num_edges
        r = (get_largest_segment_fracs(data) > 0.5).mean()
        return r, data.num_edges

    def _objective(self, radius: float) -> tuple[float, int | float]:
        """Objective function for optuna."""
        vals = []
        for mo in self._model_output:
            _v, _n_edges = self._objective_single(mo, radius)
            if math.isnan(_v):
                return float("nan"), float("nan")
            vals.append((_v, _n_edges))
        v, n_edges = np.array(vals).mean(axis=0)
        self.logger.debug("Radius %f -> 50-segment: %f, edges: %f", radius, v, n_edges)
        self._results[radius] = v, n_edges
        self._update_search_range()
        return v, n_edges

    def _update_search_range(self):
        for r in sorted(self._results):
            v = self._results[r][0]
            if v > max(self._targets):
                self._clip_radius_range(max_radius=r, reason="overachieving")
            larger_rs = [_r for _r in sorted(self._results) if _r > r]
            larger_r_vs = [self._results[_r][0] for _r in larger_rs]
            if larger_r_vs and max(larger_r_vs) > v and v < min(self._targets):
                self._clip_radius_range(min_radius=r, reason="underarchieving")
            if larger_r_vs and larger_r_vs[0] < v:
                self._clip_radius_range(max_radius=larger_rs[0], reason="decreasing")

    def _clip_radius_range(
        self,
        min_radius: float | None = None,
        max_radius: float | None = None,
        reason="",
    ):
        if min_radius is not None and min_radius > self._radius_range[0]:
            self.logger.debug("Updated min radius to %f (%s)", min_radius, reason)
            self._radius_range[0] = min_radius
        if max_radius is not None and max_radius < self._radius_range[1]:
            self.logger.debug("Updated max radius to %f (%s)", max_radius, reason)
            self._radius_range[1] = max_radius

    def _get_arrays(self):
        search_space = np.array(sorted(self._results))
        results = np.array([self._results[r] for r in search_space])
        return search_space, results

    def _get_next_radius(self) -> float:
        if self._start_radii:
            return min(
                max(self._radius_range[0], self._start_radii.pop()),
                self._radius_range[1],
            )
        search_space = np.array(
            sorted(
                self._radius_range
                + [
                    r
                    for r in self._results
                    if self._radius_range[0] < r < self._radius_range[1]
                ]
            )
        )
        if len(search_space) < 2:
            # Don't just take the middle, because else we'll never get
            # more than one
            return np.random.uniform(*self._radius_range)
        if (search_space[1:] / search_space[:-1]).max() < 1.05:
            self.logger.warning("Already very finely sampled. Abort.")
            return -1
        distances = search_space[1:] - search_space[:-1]
        max_distance = distances.max()
        return search_space[np.argmax(distances)] + max_distance / 2

    def __call__(self):
        t = Timer()
        n_sampled = 0
        while n_sampled < self._n_trials:
            radius = float(self._get_next_radius())
            if radius < 0:
                break
            v, _ = self._objective(radius)
            if not math.isnan(v):
                n_sampled += 1
        elapsed = t()
        self.logger.info("Finished radius scan in %ds.", elapsed)

        search_space, results = self._get_arrays()

        return RSResults(
            search_space=search_space,
            results=results,
            targets=self._targets,
        )
