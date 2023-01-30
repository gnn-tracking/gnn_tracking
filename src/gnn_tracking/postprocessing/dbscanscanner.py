from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.cluster import DBSCAN

from gnn_tracking.metrics.cluster_metrics import common_metrics
from gnn_tracking.postprocessing.clusterscanner import (
    AbstractClusterHyperParamScanner,
    ClusterHyperParamScanner,
    ClusterScanResult,
)

__all__ = ["DBSCANHyperParamScanner", "dbscan_scan"]

from gnn_tracking.utils.log import logger


def dbscan(graphs: np.ndarray, eps=0.99, min_samples=1) -> np.ndarray:
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

        For a convenience wrapper, take a look at `dbscan_scan`.

        Args:
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
    graphs: list[np.ndarray],
    truth: list[np.ndarray],
    sectors: list[np.ndarray],
    pts: list[np.ndarray],
    reconstructable: list[np.ndarray],
    *,
    n_jobs=1,
    n_trials: int | Callable[[int], int] = 100,
    guide="v_measure",
    epoch=None,
    start_params: dict[str, Any] | None = None,
    node_mask: list[np.ndarray] | None = None,
) -> ClusterScanResult:
    """Convenience function for scanning of DBSCAN hyperparameters to be given to
    `TCNTrainer` (see example below).

    Args:
        graphs: See ClusterHyperParamScanner
        truth: See ClusterHyperParamScanner
        sectors: See ClusterHyperParamScanner
        pts: See ClusterHyperParamScanner
        reconstructable: See ClusterHyperParamScanner
        n_jobs: Number of threads to run in parallel
        n_trials: Number of trials for optimization. If callable, it is called with the
            epoch number and should return the number of trials.
            Example: ``lambda epoch: 1 if epoch > 5 and epoch % 2 == 0 else 100`` will
            only scan HPs for every second epoch after epoch 5.
        guide: See ClusterHyperParamScanner
        epoch: Epoch that is currently being processed
        start_params: Start here
        node_mask: See ClusterHyperParamScanner

    Returns:
        ClusterScanResult

    Usage example with `TCNTrainer`::

        from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan

        faster_dbscan_scan = partial(
            dbscan_scan,
            n_jobs=12,
            n_trials=lambda epoch: 1 if epoch > 5 and epoch % 2 == 0 else 100,
        )

        trainer = TCNTrainer(
            ...,
            cluster_functions={"dbscan": faster_dbscan_scan},
        )
    """
    if start_params is None:
        start_params = {
            "eps": 0.95,
            "min_samples": 1,
        }
    if n_jobs == 1:
        logger.warning("Only using 1 thread for DBSCAN scan")
    dbss = DBSCANHyperParamScanner(
        data=graphs,
        truth=truth,
        sectors=sectors,
        pts=pts,
        reconstructable=reconstructable,
        guide=guide,
        metrics=common_metrics,
        node_mask=node_mask,
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
