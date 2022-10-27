from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.cluster import DBSCAN

from gnn_tracking.postprocessing.cluster_metrics import common_metrics
from gnn_tracking.postprocessing.clusterscanner import (
    AbstractClusterHyperParamScanner,
    ClusterHyperParamScanner,
    ClusterScanResult,
)

__all__ = ["DBSCANHyperParamScanner", "dbscan_scan"]

from gnn_tracking.utils.log import logger


def dbscan(graphs: np.ndarray, eps, min_samples) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(graphs)


class DBSCANHyperParamScanner(AbstractClusterHyperParamScanner):
    def __init__(
        self,
        *,
        eps_range: tuple[float, float] = (1e-5, 1.0),
        min_samples_range: tuple[int, int] = (1, 3),
        **kwargs,
    ):
        """Class to scan hyperparameters of DBSCAN.

        Args:
            graphs: See ClusterHyperParamScanner
            truth: See ClusterHyperParamScanner
            sectors: See ClusterHyperParamScanner
            metric: See ClusterHyperParamScanner
            eps_range: Range of epsilons to sample from
            min_samples_range: Range of min_samples to sample from
            **kwargs: Passed on to ClusterHyperParamScanner.
        """

        def suggest(trial):
            eps = trial.suggest_float("eps", *eps_range)
            min_samples = trial.suggest_int("min_samples", *min_samples_range)
            return dict(eps=eps, min_samples=min_samples)

        self.chps = ClusterHyperParamScanner(
            algorithm=dbscan,
            suggest=suggest,
            **kwargs,
        )

    def _scan(self, *args, **kwargs) -> ClusterScanResult:
        return self.chps._scan(*args, **kwargs)


def dbscan_scan(
    graphs: np.ndarray,
    truth: np.ndarray,
    sectors: np.ndarray,
    *,
    n_jobs=1,
    n_trials: int | Callable[[int], int] = 100,
    guide="v_measure",
    epoch=None,
    start_params: dict[str, Any] | None = None,
) -> ClusterScanResult:
    """Convenience function for scanning of DBSCAN hyperparameters

    Args:
        graphs: See `ClusterHyperParamScanner`
        truth: See `ClusterHyperParamScanner`
        sectors: See `ClusterHyperParamScanner`
        n_jobs: Number of threads to run in parallel
        n_trials: Number of trials for optimization. If callable, it is called with the
            epoch number and should return the number of trials.
            Example: ``lambda epoch: 1 if epoch > 5 and epoch % 2 == 0 else 100`` will
            only scan HPs for every second epoch after epoch 5.
        guide: See ClusterHyperParamScanner
        epoch: Epoch that is currently being processed
        start_params: Start here
        **kwargs: Passed on to `DBSCANHyperParamScanner`

    Returns:
        ClusterScanResult
    """
    if n_jobs == 1:
        logger.warning("Only using 1 thread for DBSCAN scan")
    dbss = DBSCANHyperParamScanner(
        graphs=graphs,
        truth=truth,
        sectors=sectors,
        guide=guide,
        metrics=common_metrics,
    )
    if isinstance(n_trials, int):
        pass
    elif isinstance(n_trials, Callable):
        n_trials = n_trials(epoch)
    else:
        raise ValueError("Invalid specification of n_trials.")

    return dbss.scan(
        n_jobs=n_jobs,
        n_trials=n_trials,
        start_params=start_params,
    )
