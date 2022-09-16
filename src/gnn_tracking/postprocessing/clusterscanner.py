from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Protocol

import numpy as np
import optuna

from gnn_tracking.utils.earlystopping import no_early_stopping
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.timing import timing

metric_type = Callable[[np.ndarray, np.ndarray], float]


@dataclasses.dataclass
class ClusterScanResult:
    study: optuna.Study
    all_metrics: dict[str, float]

    @property
    def best_params(self) -> dict[str, Any]:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value


class AbstractClusterHyperParamScanner(ABC):
    @abstractmethod
    def _scan(self, **kwargs):
        pass

    def scan(self, **kwargs) -> ClusterScanResult:
        logger.info("Starting hyperparameter scan for clustering")
        with timing("Clustering hyperparameter scan"):
            return self._scan(**kwargs)


class AlgorithmType(Protocol):
    def __call__(self, graphs: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...


class ClusterHyperParamScanner(AbstractClusterHyperParamScanner):
    def __init__(
        self,
        *,
        algorithm: AlgorithmType,
        suggest: Callable[[optuna.trial.Trial], dict[str, Any]],
        graphs: list[np.ndarray],
        truth: list[np.ndarray],
        guiding_metric: metric_type,
        sectors: list[np.ndarray] | None = None,
        cheap_guiding_metric: metric_type | None = None,
        extra_metrics: dict[str, metric_type] | None = None,
        early_stopping=no_early_stopping,
    ):
        """Class to scan hyperparameters of a clustering algorithm.

        Args:
            algorithm: Takes graph and keyword arguments
            suggest: Function that suggest parameters to optuna
            graphs:
            truth: Truth labels for clustering
            guiding_metric: Expensive metric that is taken as a figure of merit for the
                overall performance: Callable that takes truth and predicted labels
            sectors: List of 1D arrays of sector indices (answering which sector each
                hit from each graph belongs to). If None, all hits are assumed to be
                from the same sector.
            cheap_guiding_metric: Faster proxy for guiding metric
            early_stopping: Instance that can be called and has a reset method

        Example:
            # Note: This is also pre-implemented in dbscanner.py

            from sklearn import metrics
            from sklearn.cluster import DBSCAN

            def dbscan(graph, eps, min_samples):
                return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(graph)

            def suggest(trial):
                eps = trial.suggest_float("eps", 1e-5, 1.0)
                min_samples = trial.suggest_int("min_samples", 1, 50)
                return dict(eps=eps, min_samples=min_samples)

            chps = ClusterHyperParamScanner(
                dbscan,
                suggest,
                graphs,
                truths,
                expensive_metric,
                cheap_metric=metrics.v_measure_score,
            )
            study = chps.scan(n_trials=100)
            print(study.best_params)
        """
        self.algorithm = algorithm
        self.suggest = suggest
        assert [len(g) for g in graphs] == [len(t) for t in truth]
        self.graphs: list[np.ndarray] = graphs
        self.truth: list[np.ndarray] = truth
        if extra_metrics is None:
            extra_metrics = {}
        self.extra_metrics: dict[str, metric_type] = extra_metrics
        if sectors is None:
            self.sectors: list[np.ndarray] = [np.ones(t, dtype=int) for t in self.truth]
        else:
            assert [len(s) for s in sectors] == [len(t) for t in truth]
            self.sectors = sectors
        self._es = early_stopping
        self._study = None
        self._cheap_metric = cheap_guiding_metric
        self._expensive_metric = guiding_metric
        self._graph_to_sector: dict[int, int] = {}

    def _get_sector_to_study(self, i_graph: int):
        """Return index of sector to study for graph $i_graph"""
        try:
            return self._graph_to_sector[i_graph]
        except KeyError:
            pass
        available: list[int] = np.unique(self.sectors[i_graph]).tolist()  # type: ignore
        try:
            available.remove(-1)
        except ValueError:
            pass
        choice = np.random.choice(available).item()
        self._graph_to_sector[i_graph] = choice
        return choice

    def _objective(self, trial: optuna.trial.Trial) -> float:
        params = self.suggest(trial)
        cheap_foms = []
        all_labels = []
        # Do a first run, looking only at the cheap metric, stopping early
        for i_graph, (graph, truth) in enumerate(zip(self.graphs, self.truth)):
            # Consider a random sector for each graph, but keep the sector consistent
            # between different trials.
            sector = self._get_sector_to_study(i_graph)
            sector_mask = self.sectors[i_graph] == sector
            graph = graph[sector_mask]
            truth = truth[sector_mask]
            labels = self.algorithm(graph, **params)
            all_labels.append(labels)
            maybe_cheap_metric: metric_type = (
                self._cheap_metric or self._expensive_metric
            )
            cheap_foms.append(maybe_cheap_metric(truth, labels))
            if i_graph >= 2:
                v = np.nanmean(cheap_foms).item()
                trial.report(v, i_graph)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if self._cheap_metric is None:
            # What we just evaluated is actually already the expensive metric
            expensive_foms = cheap_foms
        else:
            expensive_foms = []
            # If we haven't stopped early, do a second run, looking at the expensive
            # metric
            for i_labels, (labels, truth) in enumerate(zip(all_labels, self.truth)):
                expensive_fom = self._expensive_metric(truth, labels)
                expensive_foms.append(expensive_fom)
                if i_labels >= 2:
                    trial.report(
                        np.nanmean(expensive_foms).item(), i_labels + len(self.graphs)
                    )
                if trial.should_prune():
                    raise optuna.TrialPruned()
        global_fom = np.nanmean(expensive_foms).item()
        if self._es(global_fom):
            logger.info("Stopped early")
            trial.study.stop()
        return global_fom

    @property
    def _all_metrics(self) -> dict[str, metric_type]:
        """List of all metrics with names"""
        ms = {
            "Expensive guide": self._expensive_metric,
            **self.extra_metrics,
        }
        if self._cheap_metric is not None:
            ms["Cheap guide"] = self._cheap_metric
        return ms

    def _evaluate(self) -> dict[str, float]:
        """Evaluate all metrics (on all sectors and given graphs) for the best
        parameters that we just found."""
        params = self._study.best_params
        metric_values = defaultdict(list)
        for graph, truth, sectors in zip(self.graphs, self.truth, self.sectors):
            available_sectors: list[int] = np.unique(sectors).tolist()  # type: ignore
            try:
                available_sectors.remove(-1)
            except ValueError:
                pass
            for sector in available_sectors:
                sector_mask = sectors == sector
                sector_graph = graph[sector_mask]
                sector_truth = truth[sector_mask]
                labels = self.algorithm(sector_graph, **params)
                for name, metric in self._all_metrics.items():
                    metric_values[name].append(metric(sector_truth, labels))
        return {k: np.nanmean(v).item() for k, v in metric_values.items() if v}

    def _scan(self, **kwargs) -> ClusterScanResult:
        self._es.reset()
        if self._study is None:
            self._study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(),
                direction="maximize",
            )
        assert self._study is not None  # for mypy
        self._study.optimize(
            self._objective,
            **kwargs,
        )
        return ClusterScanResult(study=self._study, all_metrics=self._evaluate())
