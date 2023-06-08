from __future__ import annotations

import math
import typing
from functools import cached_property

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from torch import Tensor
from torch_cluster import radius_graph
from torch_geometric.data import Data

from gnn_tracking.analysis.graphs import get_largest_segment_fracs
from gnn_tracking.utils.dictionaries import pivot_record_list
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.timing import Timer


def construct_graph(mo: dict[str, typing.Any], radius, max_num_neighbors=128) -> Data:
    """Construct radius graph

    Args:
        mo: Model output
        radius: Radius for radius graph
        max_num_neighbors: Maximum number of edges per node (after that: random
            sampling)
    """
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
        results: dict[float, dict[str, float]],
        targets: typing.Sequence[float],
    ):
        """This object holds the results of scanning the radius space. It performs
        interpolation to get the figures of merit (FOMs).

        Args:
            results: The results of the scan: (50%-segment fraction, n_edges)
            targets: The targets 50%-segment fractions that we're interested in
        """
        values = pivot_record_list(list(results.values()))
        df = pd.DataFrame(values)
        df["radius"] = list(results.keys())
        self.df = df.sort_values("radius")
        self.targets = targets

    def get_foms(self) -> dict[str, float]:
        foms = {}
        for t in self.targets:
            fat = self._get_foms_at_target(t)
            foms[f"n_edges_frac_segment50_{t*100:.0f}"] = fat["n_edges"]
            foms[f"n_edges_frac_segment50_{t*100:.0f}_r"] = fat["radius"]
            foms[f"frac75_at_frac_segment50_{t*100:.0f}"] = fat["frac75"]
            foms[f"frac100_at_frac_segment50_{t*100:.0f}"] = fat["frac100"]
        idx_max_frac50 = self.df["frac50"].argmax()
        fat = self.df.iloc[idx_max_frac50]
        foms["max_frac_segment50"] = fat["frac50"]
        foms["n_edges_max_frac_segment50"] = fat["n_edges"]
        foms["max_frac_segment50_r"] = fat["radius"]
        foms["frac75_at_max_frac_segment50"] = fat["frac75"]
        foms["frac100_at_max_frac_segment50"] = fat["frac100"]
        return foms

    def plot(self) -> plt.Axes:
        """Plot interpolation"""
        bounds = (
            self.df["radius"].min(),
            self.df["radius"].max(),
        )
        xs = np.linspace(*bounds, 1000)
        df = pd.DataFrame(pivot_record_list([self._eval_spline(x) for x in xs]))
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot("radius", "frac50", data=df, marker="none", color="C0", label="frac 50")
        ax.plot("radius", "frac50", data=self.df, marker="o", color="C0")
        ax2.plot("radius", "n_edges", data=df, marker="none", color="C1", label="edges")
        ax2.plot("radius", "n_edges", data=self.df, marker="o", color="C1")
        ax.plot("radius", "frac75", data=df, marker="none", color="C2", label="frac 75")
        ax.plot("radius", "frac75", data=self.df, marker="o", color="C2")
        ax.plot(
            "radius", "frac100", data=df, marker="none", color="C3", label="frac 100"
        )
        ax.plot("radius", "frac100", data=self.df, marker="o", color="C3")
        for t in self.targets:
            ax.axhline(t, linestyle="--", lw=1, color="C0")
        for target in self.targets:
            ax.axvline(
                self._get_target_radius(target), linestyle="--", lw=1, color="C0"
            )
        fig.legend()
        return ax

    @cached_property
    def _spline(self):
        return CubicSpline(self.df["radius"], self.df)

    def _eval_spline(self, radius: float) -> dict[str, float]:
        # Unclear why sometimes the spline returns a 2D array
        _r = self._spline(radius).squeeze().tolist()
        return dict(zip(self.df.columns, _r))

    def _get_target_radius(self, target: float) -> float:
        """Radius at which the 50%-segment fraction = target"""
        if target > self.df["frac50"].max():
            return float("nan")
        bounds = (
            self.df["radius"].min().item(),
            self.df["radius"].max().item(),
        )
        initial_value = sum(bounds) / 2
        return minimize(
            lambda radius: np.abs(self._eval_spline(radius)["frac50"] - target),
            x0=initial_value,
            bounds=(bounds,),
        ).x.item()

    def _get_foms_at_target(self, target: float) -> dict[str, float]:
        _nan_results = {k: float("nan") for k in self.df.columns}
        if len(self.df) < 2:
            return _nan_results
        target_r = self._get_target_radius(target)
        if math.isnan(target_r):
            return _nan_results
        return self._eval_spline(target_r)


class ComputationAborted(Exception):
    pass


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
        """Scan different radii for the radius graph.

        .. code-block:: python

            rs = RadiusScanner(...)
            rs_results = rs()
            foms = rs_results.get_foms()

        Args:
            model_output:
            radius_range: Range of radii to scan
            max_num_neighbors: Parameter to torch scatter radius graph: Maximal number
                of neighbors to consider
            n_trials:
            max_edges: Maximal number of edges for which we still evaluate the metrics
            target_fracs: Target fractions of 50%-segments that we are interested in
            start_radii: First guesses for radii
        """
        if start_radii is None:
            start_radii = []
        self._start_radii = start_radii
        self._model_output = model_output
        self._radius_range = list(radius_range)
        self._max_num_neighbors = max_num_neighbors
        self._n_trials = n_trials
        self.logger = get_logger("RadiusHP")
        self._max_edges = max_edges
        self._targets = target_fracs
        self._results: dict[float, dict[str, float | int]] = {}

    def __call__(self) -> RSResults:
        """Run radius scan"""
        t = Timer()
        n_sampled = 0
        while n_sampled < self._n_trials:
            radius = float(self._suggest_radius())
            if radius < 0:
                break
            try:
                self._objective(radius)
            except ComputationAborted:
                continue
            n_sampled += 1
        elapsed = t()
        self.logger.info("Finished radius scan in %ds.", elapsed)

        return RSResults(
            results=self._results,
            targets=self._targets,
        )

    def _objective_single_point_cloud(
        self, mo: dict[str, typing.Any], radius: float
    ) -> dict[str, float | int]:
        """Construct graph for given point cloud and radius

        Returns:
            50%-segment fraction, number of edges. nan is returned for both if
            we exceed maximal number of edges.
        """
        if radius > self._radius_range[1]:
            raise ComputationAborted("Exceeded radius range")
        data = construct_graph(
            mo, radius=radius, max_num_neighbors=self._max_num_neighbors
        )
        if data.num_edges > self._max_edges:
            self._clip_radius_range(max_radius=radius, reason="too many edges")
            raise ComputationAborted("Too many edges")
        frac50 = (get_largest_segment_fracs(data) > 0.5).mean().item()
        frac75 = (get_largest_segment_fracs(data) > 0.75).mean().item()
        frac100 = (get_largest_segment_fracs(data) == 1).mean().item()
        return {
            "frac50": frac50,
            "frac75": frac75,
            "frac100": frac100,
            "n_edges": data.num_edges,
        }

    def _objective(self, radius: float) -> dict[str, float | int]:
        """Construct graphs for all validation point clouds at a given radius and
        averages the results.

        Returns:
            50%-segment fraction, number of edges. nan is returned for both if
            we exceed maximal number of edges.
        """
        vals = pivot_record_list(
            [
                self._objective_single_point_cloud(mo, radius)
                for mo in self._model_output
            ]
        )
        vals_avg = {k: np.array(v).mean().item() for k, v in vals.items()}
        self.logger.debug(
            "Radius %f -> 50-segment: %f, edges: %f",
            radius,
            vals_avg["frac50"],
            vals_avg["n_edges"],
        )
        self._results[radius] = vals_avg
        self._update_search_range()
        return vals_avg

    def _update_search_range(self):
        """Update search range based on all previous results"""
        for r in sorted(self._results):
            v = self._results[r]["frac50"]
            if v > max(self._targets):
                self._clip_radius_range(max_radius=r, reason="overachieving")
            larger_rs = [_r for _r in sorted(self._results) if _r > r]
            larger_r_vs = [self._results[_r]["frac50"] for _r in larger_rs]
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
        """Clip radius range to given values

        Args:
            min_radius: If not None, clip min radius to this value
            max_radius: If not None, clip max radius to this value
            reason: Reason for clipping
        """
        if min_radius is not None and min_radius > self._radius_range[0]:
            self.logger.debug("Updated min radius to %f (%s)", min_radius, reason)
            self._radius_range[0] = min_radius
        if max_radius is not None and max_radius < self._radius_range[1]:
            self.logger.debug("Updated max radius to %f (%s)", max_radius, reason)
            self._radius_range[1] = max_radius

    def _get_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get arrays of radii and results"""
        search_space = np.array(sorted(self._results))
        results = np.array([self._results[r] for r in search_space])
        return search_space, results

    def _suggest_radius(self) -> float:
        """Get next radius to evaluate"""
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
