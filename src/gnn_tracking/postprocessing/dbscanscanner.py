from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from gnn_tracking.postprocessing.clusterscanner import (
    AbstractClusterHyperParamScanner,
    ClusterHyperParamScanner,
    ClusterScanResult,
)

__all__ = ["DBSCANHyperParamScanner"]


def dbscan(graph: np.ndarray, eps, min_samples) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(graph)


class DBSCANHyperParamScanner(AbstractClusterHyperParamScanner):
    def __init__(
        self,
        *,
        eps_range: tuple[float, float] = (1e-5, 1.0),
        min_samples_range: tuple[int, int] = (1, 50),
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

    def _scan(self, **kwargs) -> ClusterScanResult:
        return self.chps._scan(**kwargs)
