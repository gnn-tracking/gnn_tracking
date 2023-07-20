import copy
import math
import typing
from functools import cached_property

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pytorch_lightning.core.mixins import HyperparametersMixin
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

from gnn_tracking.analysis.graphs import get_largest_segment_fracs
from gnn_tracking.models.graph_construction import knn_with_max_radius
from gnn_tracking.utils.dictionaries import pivot_record_list
from gnn_tracking.utils.log import logger

# ruff: noqa: ARG002


class KScanResults:
    def __init__(
        self,
        results: pd.DataFrame,
        targets: typing.Sequence[float],
    ):
        """This object holds the results of scanning over ks. It performs
        interpolation to get the figures of merit (FOMs).

        Args:
            results: The results of the scan: (k, n_edges, frac50, ...)
            targets: The targets 50%-segment fractions that we're interested in
        """
        self.df = results.sort_values("k")
        self.df["k"] = self.df.index
        self.targets = targets

    def get_foms(self) -> dict[str, float]:
        foms = {}
        for t in self.targets:
            fat = self._get_foms_at_target(t)
            foms[f"n_edges_frac_segment50_{t*100:.0f}"] = fat["n_edges"]
            foms[f"n_edges_frac_segment50_{t*100:.0f}_k"] = fat["k"]
            foms[f"frac75_at_frac_segment50_{t*100:.0f}"] = fat["frac75"]
            foms[f"frac100_at_frac_segment50_{t*100:.0f}"] = fat["frac100"]
        idx_max_frac50 = self.df["frac50"].argmax()
        fat = self.df.iloc[idx_max_frac50]
        foms["max_frac_segment50"] = fat["frac50"]
        foms["n_edges_max_frac_segment50"] = fat["n_edges"]
        foms["max_frac_segment50_k"] = fat["k"]
        foms["frac75_at_max_frac_segment50"] = fat["frac75"]
        foms["frac100_at_max_frac_segment50"] = fat["frac100"]
        return foms

    def plot(self) -> plt.Axes:
        """Plot interpolation"""
        bounds = (
            self.df["k"].min(),
            self.df["k"].max(),
        )
        xs = np.linspace(*bounds, 1000)
        df = pd.DataFrame(pivot_record_list([self._eval_spline(x) for x in xs]))
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot("k", "frac50", data=df, marker="none", color="C0", label="frac 50")
        ax.plot("k", "frac50", data=self.df, marker="o", color="C0", ls="none")
        ax2.plot("k", "n_edges", data=df, marker="none", color="C1", label="edges")
        ax2.plot("k", "n_edges", data=self.df, marker="o", color="C1", ls="none")
        ax.plot("k", "frac75", data=df, marker="none", color="C2", label="frac 75")
        ax.plot("k", "frac75", data=self.df, marker="o", color="C2", ls="none")
        ax.plot("k", "frac100", data=df, marker="none", color="C3", label="frac 100")
        ax.plot("k", "frac100", data=self.df, marker="o", color="C3", ls="none")
        for t in self.targets:
            ax.axhline(t, linestyle="--", lw=1, color="C0")
        for target in self.targets:
            ax.axvline(self._get_target_k(target), linestyle="--", lw=1, color="C0")
        fig.legend(loc="lower right")
        return ax

    @cached_property
    def _spline(self):
        return CubicSpline(self.df["k"], self.df)

    def _eval_spline(self, k: float) -> dict[str, float]:
        # Unclear why sometimes the spline returns a 2D array
        _r = self._spline(k).squeeze().tolist()
        return dict(zip(self.df.columns, _r))

    def _get_target_k(self, target: float) -> float:
        """K at which the 50%-segment fraction = target"""
        if target > self.df["frac50"].max():
            return float("nan")
        bounds = (
            self.df["k"].min().item(),
            self.df["k"].max().item(),
        )
        initial_value = sum(bounds) / 2
        return minimize(
            lambda k: np.abs(self._eval_spline(k)["frac50"] - target),
            x0=initial_value,
            bounds=(bounds,),
        ).x.item()

    def _get_foms_at_target(self, target: float) -> dict[str, float]:
        _nan_results = {k: float("nan") for k in self.df.columns}
        if len(self.df) < 2:
            return _nan_results
        target_r = self._get_target_k(target)
        if math.isnan(target_r):
            return _nan_results
        return self._eval_spline(target_r)


_DEFAULT_KS = np.arange(0, 40, 4).tolist()


class GraphConstructionKNNScanner(HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        ks: list[int] = _DEFAULT_KS,
        *,
        targets=(0.8, 0.85, 0.88, 0.9, 0.93, 0.95, 0.97, 0.99),
        max_radius=1.0,
        pt_thld=0.9,
        subsample_pids: int | None = None,
        max_edges=5_000_000,
    ):
        """Scan over different values of k to build a graph and calculate the figures
        of merit.

        Args:
            ks: List of ks to scan. Results will be interpolated between these values,
                so it's a good idea to not make them too dense.
            targets: Targets for the 50%-segment fraction that we aim for (we will find
                the k that gets us closest to these targets and report the number
                of edges for these places). Does not impact compute time.
            max_radius: Maximum length of edges for the KNN graph.
            pt_thld: pt threshold for evaluation of the 50%-segment fraction.
            subsample_pids: Set to a number to subsample the number of pids in the
                evaluation of the 50%-segment fraction. This is useful for speeding
                up the evaluation of the 50%-segment fraction, but it will lead to
                a less accurate result/statistical fluctuations.
            max_edges: Do not attempt to compute metrics for more than this number of
                edges in the knn graph
        """
        super().__init__()
        self.save_hyperparameters()
        self._results = []

    @property
    def results_raw(self) -> pd.DataFrame:
        """DataFrame with raw results for all graphs and all k"""
        return pd.DataFrame.from_records(self._results)

    def get_results(self) -> KScanResults:
        """Get results object"""
        mean_results = self.results_raw.groupby("k").mean()
        return KScanResults(mean_results, targets=self.hparams.targets)

    def get_foms(self) -> dict[str, float]:
        """Get figures of merit (convenience method that uses the appropriate method
        of `KSCanResults`)."""
        return self.get_results().get_foms()

    def reset(self):
        """Reset the results. Will be automatically called every time we run on
        a batch with `i_batch == 0`.
        """
        self._results = []

    def __call__(
        self, data: Data, i_batch: int, *, progress=False, latent: Tensor | None = None
    ) -> None:
        """Run on graph

        Args:
            data: Data object. `data.x` is the space used for clustering
            i_batch: Batch number. Will reset saved data for `i_batch == 0`.
            progress: Show progress bar
            latent: Use this instead of `data.x`

        Returns:
            None
        """
        if i_batch == 0:
            self.reset()
        iterator = self.hparams.ks
        data = copy.copy(data.detach())
        if latent is not None:
            data.x = latent
        if progress:
            iterator = tqdm(iterator)
        for k in iterator:
            r = self._evaluate_graph(data, k)
            if r is None:
                break
            self._results.append(r)

    def _evaluate_graph(self, data: Data, k: int) -> dict[str, float] | None:
        """Evaluate metrics for single graphs

        Args:
            data:
            k:

        Returns:
            None if computation was aborted
        """
        data.edge_index = knn_with_max_radius(
            data.x, k=k, max_radius=self.hparams.max_radius
        )
        n_edges = data.edge_index.shape[1]
        if n_edges > self.hparams.max_edges:
            msg = (
                f"Not scanning k>={k} because max edges exceeded "
                f"({n_edges} > {self.hparams.max_edges})"
            )
            logger.warning(msg)
            return None
        data.y = (
            data.particle_id[data.edge_index[0]] == data.particle_id[data.edge_index[1]]
        )
        lsfs = get_largest_segment_fracs(data)
        return {
            "k": k,
            "frac50": (lsfs > 0.5).mean().item(),
            "frac75": (lsfs > 0.75).mean().item(),
            "frac100": (lsfs == 1).mean().item(),
            "n_edges": n_edges,
        }
