import numpy as np
from pytorch_lightning.core.mixins import HyperparametersMixin
from sklearn.cluster import DBSCAN

from gnn_tracking.metrics.cluster_metrics import common_metrics
from gnn_tracking.postprocessing.clusterscanner import (
    ClusterHyperParamScanner,
    ClusterScanResult,
)


def dbscan(graphs: np.ndarray, eps=0.99, min_samples=1) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(graphs)


class DBSCANHyperParamScanner(HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        eps_range: tuple[float, float] = (1e-5, 1.0),
        min_samples_range: tuple[int, int] = (1, 3),
        n_trials: int = 10,
        n_jobs: int = 1,
    ):
        """Class to scan hyperparameters of DBSCAN.

        For a convenience wrapper, take a look at `dbscan_scan`.

        Args:
            eps_range: Range of epsilons to sample from
            min_samples_range: Range of min_samples to sample from
            **kwargs: Passed on to ClusterHyperParamScanner.
        """
        super().__init__()
        self.save_hyperparameters()

    def __call__(self, epoch=None, start_params=None, **kwargs) -> ClusterScanResult:
        def suggest(trial):
            eps = trial.suggest_float("eps", *self.hparams.eps_range)
            min_samples = trial.suggest_int(
                "min_samples", *self.hparams.min_samples_range
            )
            return {"eps": eps, "min_samples": min_samples}

        chps = ClusterHyperParamScanner(
            algorithm=dbscan,
            suggest=suggest,
            guide="v_measure",
            metrics=common_metrics,
            **kwargs,
        )
        return chps.scan(
            start_params=start_params,
            n_trials=self.hparams.n_trials,
            n_jobs=self.hparams.n_jobs,
        )
