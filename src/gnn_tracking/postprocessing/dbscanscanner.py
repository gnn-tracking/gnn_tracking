import math

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from torch import Tensor as T
from torch_geometric.data import Data
from tqdm import tqdm

from gnn_tracking.metrics.cluster_metrics import (
    flatten_track_metrics,
    tracking_metric_df,
    tracking_metrics,
)
from gnn_tracking.postprocessing.clusterscanner import ClusterScanner
from gnn_tracking.postprocessing.fastrescanner import DBSCANFastRescan
from gnn_tracking.utils.dictionaries import add_key_prefix


def dbscan(graphs: np.ndarray, eps=0.99, min_samples=1) -> np.ndarray:
    """Convenience wrapper around `sklearn`'s DBSCAN implementation."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(graphs)


class OCScanResults:
    _PARAMETERS = ["eps", "min_samples"]

    def __init__(self, df: pd.DataFrame):
        """Restults of `DBSCANHyperparamScanner` and friends."""
        self._df = df
        gb = self.df.groupby(self._PARAMETERS)
        _df_mean = gb.mean()
        _df_std = gb.std() / math.sqrt(len(_df_mean))
        self._df_mean = _df_mean.merge(
            _df_std,
            left_on=self._PARAMETERS,
            right_on=self._PARAMETERS,
            suffixes=("", "_std"),
        )
        self._df_mean.reset_index(inplace=True)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def df_mean(self) -> pd.DataFrame:
        """Mean and std grouped by hyperparameters."""
        return self._df_mean

    def get_foms(self, guide="double_majority_pt0.9") -> dict[str, float]:
        """Get figures of merit"""
        fom_cols = [col for col in self._df_mean if col not in self._PARAMETERS]
        assert guide in fom_cols
        best_idx = self._df_mean[guide].idxmax()
        best_series = self._df_mean.iloc[best_idx]
        foms = add_key_prefix(best_series[fom_cols].to_dict(), "trk.")
        for param in self._PARAMETERS:
            foms[f"best_dbscan_{param}"] = best_series[param]
        return foms

    def get_n_best_trials(
        self, n: int, guide="double_majority_pt0.9"
    ) -> list[dict[str, float]]:
        return (
            self._df_mean.sort_values(guide, ascending=False)
            .head(n)[self._PARAMETERS]
            .to_dict(orient="records")
        )


class DBSCANHyperParamScanner(ClusterScanner):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        eps_range=(0, 1),
        min_samples_range=(1, 4),
        n_trials=10,
        keep_best=0,
        n_jobs: int | None = None,
        guide: str = "double_majority_pt0.9",
        pt_thlds=(0.0, 0.5, 0.9, 1.5),
        max_eta: float = 4.0,
    ):
        """Scan for hyperparameters of DBSCAN. Use this scanner for validation.
        Even with few trials, it will eventually apply finer samples to the best
        region, because it will keep the best trials from the previous epoch
        (make sure th choose non-zero ``kep_best``).

        Args:
            eps_range: Range of DBSCAN radii to scan
            min_samples_range: Range (INCLUSIVE!) of minimum number of samples for
                DBSCAN
            n_trials: Total number of trials
            keep_best: Keep this number of the best `(eps, min_samples)` pairs from
                the current epoch and make sure to scan over them again in the next
                epoch.
            n_jobs: Number of jobs to use for parallelization
            guide: Report tracking metrics for parameters that maximize this metric
            pt_thlds: list of pT thresholds for the tracking metrics
            max_eta: Max eta for tracking metrics
        """
        super().__init__()
        self.save_hyperparameters()
        # todo: this is for backwards compatibility, remove in future
        self.hparams.guide = self.hparams.guide.removeprefix("trk.")
        self._results = []
        self._rng = np.random.default_rng()
        self._trials = []
        self._rng = np.random.default_rng()
        self.reset()

    def get_results(self) -> OCScanResults:
        return OCScanResults(pd.DataFrame.from_records(self._results))

    def get_foms(self) -> dict[str, float]:
        return self.get_results().get_foms()

    def _get_best_trials(self) -> list[dict[str, float]]:
        if not self._results:
            return []
        return self.get_results().get_n_best_trials(self.hparams.keep_best)

    def _reset_trials(self) -> None:
        self._trials = self._get_best_trials()
        size_random = self.hparams.n_trials - len(self._trials)
        eps = self._rng.uniform(*self.hparams.eps_range, size=size_random)
        min_samples = self._rng.integers(
            self.hparams.min_samples_range[0],
            self.hparams.min_samples_range[1] + 1,
            size=size_random,
        )
        self._trials = [{"eps": e, "min_samples": n} for e, n in zip(eps, min_samples)]

    def reset(self):
        """Reset the results. Will be automatically called every time we run on
        a batch with `i_batch == 0`.
        """
        self._reset_trials()
        self._best_trials = []
        self._results = []

    def __call__(
        self,
        data: Data,
        out: dict[str, T],
        i_batch: int,
        *,
        progress=False,
    ):
        if (ec_hist_mask := out.get("ec_hit_mask")) is not None:
            if not ec_hist_mask.all():
                raise NotImplementedError(
                    "Handling of orphan node pruning not implemented"
                )
        if i_batch == 0:
            self.reset()
        scanner = DBSCANFastRescan(
            out["H"].detach().cpu().numpy(),
            max_eps=max(v["eps"] for v in self._trials),
            n_jobs=self.hparams.n_jobs,
        )
        iterator = self._trials
        if progress:
            iterator = tqdm(iterator)
        for trial in iterator:
            labels = scanner.cluster(eps=trial["eps"], min_pts=trial["min_samples"])
            metrics = tracking_metrics(
                truth=data.particle_id.detach().cpu().numpy(),
                predicted=labels,
                pts=data.pt.detach().cpu().numpy(),
                eta=data.eta.detach().cpu().numpy(),
                reconstructable=data.reconstructable.detach().cpu().numpy(),
                pt_thlds=self.hparams.pt_thlds,
                max_eta=self.hparams.max_eta,
            )
            self._results.append(
                {
                    "i_batch": i_batch,
                    "eps": trial["eps"],
                    "min_samples": trial["min_samples"],
                    **flatten_track_metrics(metrics),
                }
            )


class DBSCANHyperParamScannerFixed(DBSCANHyperParamScanner):
    def __init__(
        self,
        trials: list[dict[str, float]],
        *,
        n_jobs: int | None = None,
        pt_thlds=(0.0, 0.5, 0.9, 1.5),
        max_eta: float = 4.0,
    ):
        """Scan grid for hyperparameters of DBSCAN. While `DBSCANHyperParamScanner`
        is for use in validation steps, this is for use in detailed testing.

        Args:
            trials: List of trials to run
            n_jobs: Number of jobs to use for parallelization
            pt_thlds: list of pT thresholds for the tracking metrics
            max_eta: Max eta for tracking metrics
        """
        super().__init__(n_jobs=n_jobs, pt_thlds=pt_thlds, max_eta=max_eta)
        self._trials = trials

    def _reset_trials(self) -> None:
        pass


class DBSCANPerformanceDetails(DBSCANHyperParamScanner):
    def __init__(self, eps: float, min_samples: int):
        """Get information about detailed performance for fixed DBSCAN parameters.
        See `get_results` for outputs.

        Args:
            eps: DBSCAN epsilon
            min_samples: DBSCAN min_samples
        """
        super().__init__()
        self.save_hyperparameters()
        self._h_dfs = []
        self._c_dfs = []

    def __call__(self, data: Data, out: dict[str, T], i_batch: int) -> None:
        def prep(tensor):
            return tensor.detach().cpu().numpy()

        labels = dbscan(
            out["H"].detach().cpu().numpy(),
            eps=self.hparams.eps,
            min_samples=self.hparams.min_samples,
        )
        h_df = pd.DataFrame(
            {
                "c": labels,
                "id": prep(data.particle_id),
                "reconstructable": prep(data.reconstructable),
                "pt": prep(data.pt),
                "eta": prep(data.eta),
            }
        )
        c_df = tracking_metric_df(h_df)
        self._h_dfs.append(h_df)
        self._c_dfs.append(c_df)

    def get_results(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Get results

        Returns:
            Tuple of (h_dfs, c_dfs), where h_dfs is a list of dataframes with
            information about all hits and c_dfs is a list of dataframes with
            information about all clusters.
            See `tracking_metric_df` for details about the information about both
            dataframes..
        """
        return self._h_dfs, self._c_dfs

    def get_foms(self) -> dict[str, float]:
        return {}
