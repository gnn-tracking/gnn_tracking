from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import optuna

from gnn_tracking.metrics.cluster_metrics import ClusterMetricType
from gnn_tracking.utils.earlystopping import no_early_stopping
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.timing import timing


@dataclasses.dataclass
class ClusterScanResult:
    """Result of the scan over different clustering hyperparameters."""

    #: This object contains information about every single trial.
    study: optuna.Study
    #: Complete metrics of the trial that was deemed most successful
    metrics: dict[str, float]

    @property
    def best_params(self) -> dict[str, Any]:
        """Hyperparameters with the best score."""
        return self.study.best_params

    @property
    def best_value(self) -> float:
        """Best score"""
        return self.study.best_value


class AbstractClusterHyperParamScanner(ABC):
    """Abstract base class for classes that implement hyperparameter scanning of
    clustering algorithms.
    """

    @abstractmethod
    def _scan(
        self,
        start_params: dict[str, Any] | None = None,
    ) -> ClusterScanResult:
        pass

    def scan(
        self, start_params: dict[str, Any] | None = None, **kwargs
    ) -> ClusterScanResult:
        if start_params is not None:
            logger.debug("Starting from params: %s", start_params)
        logger.info("Starting hyperparameter scan for clustering")
        with timing("Clustering hyperparameter scan & metric evaluation"):
            return self._scan(**kwargs)


class ClusterAlgorithmType(Protocol):
    """Type of a clustering algorithm."""

    def __call__(self, graphs: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...


class ClusterHyperParamScanner(AbstractClusterHyperParamScanner):
    def __init__(
        self,
        *,
        algorithm: ClusterAlgorithmType,
        suggest: Callable[[optuna.trial.Trial], dict[str, Any]],
        graphs: list[np.ndarray],
        truth: list[np.ndarray],
        pts: list[np.ndarray],
        guide: str,
        metrics: dict[str, ClusterMetricType],
        sectors: list[np.ndarray] | None = None,
        guide_proxy="",
        early_stopping=no_early_stopping,
        pt_thlds: Iterable[float] = (
            0.0,
            0.5,
            0.9,
            1.5,
        ),
    ):
        """Class to scan hyperparameters of a clustering algorithm.

        Args:
            algorithm: Takes graph and keyword arguments
            suggest: Function that suggest parameters to optuna
            graphs:
            truth: Truth labels for clustering
            pts: Pt values for each graph
            guide: Name of expensive metric that is taken as a figure of merit
                for the overall performance. If the corresponding metric function
                returns a dict, the key should be key.subkey.
            metrics: Dictionary of metrics to evaluate. Each metric is a function that
                takes truth and predicted labels as numpy arrays and returns a float.
            sectors: List of 1D arrays of sector indices (answering which sector each
                hit from each graph belongs to). If None, all hits are assumed to be
                from the same sector.
            guide_proxy: Faster proxy for guiding metric. See
            early_stopping: Instance that can be called and has a reset method
            pt_thlds: Pt thresholds to be used in metric evaluation (for metrics that
                support it).

        Example::

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
                algorithm=dbscan,
                suggest=suggest,
                graphs=graphs,
                truth=truths,
                pts=pts,
                guide="v_measure_score",
                metrics=dict(v_measure_score=metrics.v_measure_score),
            )
            study = chps.scan(n_trials=100)
            print(study.best_params)
        """
        self.algorithm = algorithm
        self.suggest = suggest
        assert [len(g) for g in graphs] == [len(t) for t in truth]
        assert len(graphs) > 0
        self.graphs: list[np.ndarray] = graphs
        self.truth: list[np.ndarray] = truth
        self.pts: list[np.ndarray] = pts
        self.metrics: dict[str, ClusterMetricType] = metrics
        if sectors is None:
            self.sectors: list[np.ndarray] = [np.ones(t, dtype=int) for t in self.truth]
        else:
            assert [len(s) for s in sectors] == [len(t) for t in truth]
            self.sectors = sectors
        self._es = early_stopping
        self._study = None
        self._cheap_metric = guide_proxy
        self._expensive_metric = guide
        self._graph_to_sector: dict[int, int] = {}
        #: Number of graphs to look at before using accumulated statistics to maybe
        #: prune trial.
        self.pruning_grace_period = 20
        #: Number of trials completed
        self._n_trials_completed = 0
        #: Number of trials that were pruned
        self._n_trials_pruned = 0
        self.pt_thlds = list(pt_thlds)

    def _get_sector_to_study(self, i_graph: int):
        """Return index of sector to study for graph $i_graph.
        Takes a random one the first time, but then remembers the sector so that we
        get the same one for the same graph.
        """
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

    def _get_explicit_metric(
        self, name: str, *, predicted: np.ndarray, truth: np.ndarray, pts: np.ndarray
    ) -> float:
        """Get metric value from dict of metrics."""
        arguments = dict(
            truth=truth,
            predicted=predicted,
            pts=pts,
            pt_thlds=self.pt_thlds,
        )
        if "." in name:
            metric, _, subkey = name.partition(".")
            try:
                return self.metrics[metric](**arguments)[subkey]  # type: ignore
            except KeyError:
                pass
        return self.metrics[name](**arguments)  # type: ignore

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for optuna."""
        params = self.suggest(trial)
        cheap_foms = []
        all_labels = []
        # Do a first run, looking only at the cheap metric, stopping early
        for i_graph, (graph, truth, pts) in enumerate(
            zip(self.graphs, self.truth, self.pts)
        ):
            # Consider a random sector for each graph, but keep the sector consistent
            # between different trials.
            sector = self._get_sector_to_study(i_graph)
            sector_mask = self.sectors[i_graph] == sector
            graph = graph[sector_mask]
            truth = truth[sector_mask]
            pts = pts[sector_mask]
            labels = self.algorithm(graph, **params)
            all_labels.append(labels)
            cheap_foms.append(
                self._get_explicit_metric(
                    self._cheap_metric or self._expensive_metric,
                    truth=truth,
                    predicted=labels,
                    pts=pts,
                )
            )
            if i_graph >= self.pruning_grace_period:
                v = np.nanmean(cheap_foms).item()
                trial.report(v, i_graph)
            if trial.should_prune():
                self._n_trials_pruned += 1
                raise optuna.TrialPruned()
        if not self._cheap_metric:
            # What we just evaluated is actually already the expensive metric
            expensive_foms = cheap_foms
        else:
            expensive_foms = []
            # If we haven't stopped early, do a second run, looking at the expensive
            # metric
            for i_labels, (labels, truth, pts) in enumerate(
                zip(all_labels, self.truth, self.pts)
            ):
                sector = self._get_sector_to_study(i_labels)
                sector_mask = self.sectors[i_labels] == sector
                truth = truth[sector_mask]
                expensive_fom = self._get_explicit_metric(
                    self._expensive_metric,
                    truth=truth,
                    predicted=labels,
                    pts=pts,
                )
                expensive_foms.append(expensive_fom)
                if i_labels >= 2:
                    trial.report(
                        np.nanmean(expensive_foms).item(), i_labels + len(self.graphs)
                    )
                if trial.should_prune():
                    self._n_trials_pruned += 1
                    raise optuna.TrialPruned()
        global_fom = np.nanmean(expensive_foms).item()
        if self._es(global_fom):
            logger.info("Stopped early")
            trial.study.stop()
        self._n_trials_completed += 1
        return global_fom

    def _evaluate(self) -> dict[str, float]:
        """Evaluate all metrics (on all sectors and given graphs) for the best
        parameters that we just found.
        """
        logger.debug("Evaluating all metrics for best clustering")
        with timing("Evaluating all metrics"):
            return self.__evaluate()

    def __evaluate(self) -> dict[str, float]:
        """See _evaluate."""
        assert self._study is not None  # mypy
        params = self._study.best_params
        metric_values = defaultdict(list)
        for graph, truth, sectors, pts in zip(
            self.graphs, self.truth, self.sectors, self.pts
        ):
            available_sectors: list[int] = np.unique(sectors).tolist()  # type: ignore
            try:
                available_sectors.remove(-1)
            except ValueError:
                pass
            for sector in available_sectors:
                sector_mask = sectors == sector
                sector_graph = graph[sector_mask]
                sector_truth = truth[sector_mask]
                sector_pts = pts[sector_mask]
                labels = self.algorithm(sector_graph, **params)
                for name, metric in self.metrics.items():
                    r = metric(
                        truth=sector_truth,
                        predicted=labels,
                        pts=sector_pts,
                        pt_thlds=self.pt_thlds,
                    )
                    if not isinstance(r, Mapping):
                        metric_values[name].append(r)
                    else:
                        for k, v in r.items():
                            metric_values[f"{name}.{k}"].append(v)
        return {k: np.nanmean(v).item() for k, v in metric_values.items() if v}

    def _scan(
        self, start_params: dict[str, Any] | None = None, **kwargs
    ) -> ClusterScanResult:
        """Run the scan."""
        self._es.reset()
        if self._study is None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(),
                direction="maximize",
            )
        assert self._study is not None  # for mypy
        if start_params is not None:
            self._study.enqueue_trial(start_params)
        self._study.optimize(
            self._objective,
            **kwargs,
        )
        logger.info(
            "Completed %d trials, pruned %d trials",
            self._n_trials_completed,
            self._n_trials_pruned,
        )
        return ClusterScanResult(study=self._study, metrics=self._evaluate())
