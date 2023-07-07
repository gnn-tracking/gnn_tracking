# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

from typing import Callable, Sequence

import numpy as np
from pytorch_lightning.core.mixins import HyperparametersMixin
from sklearn.cluster import DBSCAN

from gnn_tracking.metrics.cluster_metrics import common_metrics
from gnn_tracking.postprocessing.clusterscanner import (
    ClusterHyperParamScanner,
    ClusterScanResult,
)
from gnn_tracking.utils.lightning import obj_from_or_to_hparams


def dbscan(graphs: np.ndarray, eps=0.99, min_samples=1) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(graphs)


class DBSCANHyperParamScanner(HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        eps_range: tuple[float, float] = (1e-5, 1.0),
        min_samples_range: tuple[int, int] = (1, 1),
        n_trials: int | Callable | Sequence = 10,
        n_jobs: int = 1,
        guide="trk.double_majority_pt0.9",
    ):
        """Class to scan hyperparameters of DBSCAN.

        For a convenience wrapper, take a look at `dbscan_scan`.

        Args:
            eps_range: Range of epsilons to sample from
            min_samples_range: Range of min_samples to sample from
            n_trials: Number of trials to run. If callable: Function that returns
                the number for given epoch. If sequence: Will be indexed by epoch.
            n_jobs: Number of jobs to run in parallel.
            guide: Guiding metric
        """
        super().__init__()
        self.save_hyperparameters(ignore="n_trials")
        self._n_trials = obj_from_or_to_hparams(self, "n_trials", n_trials)

    def _get_n_trials(self, epoch: int) -> int:
        if isinstance(self._n_trials, int):
            return self._n_trials
        elif isinstance(self._n_trials, Sequence):
            if len(self._n_trials) <= epoch:
                return self._n_trials[-1]
            return self._n_trials[epoch]
        return self._n_trials(epoch)

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
            guide=self.hparams.guide,
            metrics=common_metrics,
            **kwargs,
        )
        return chps.scan(
            start_params=start_params,
            n_trials=self._get_n_trials(epoch),
            n_jobs=self.hparams.n_jobs,
        )
