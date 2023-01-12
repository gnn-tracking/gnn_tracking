from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import optuna

from gnn_tracking.metrics.cluster_metrics import ClusterMetricType
from gnn_tracking.utils.earlystopping import no_early_stopping
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.timing import timing


class ClusterScanResult:
    def __init__(
        self,
        metrics: dict[str, float],
        best_params: dict[str, float],
        best_value: float,
    ):
        self.metrics = metrics

        #: Hyperparameters with best score
        self.best_params = best_params

        #: Best score
        self.best_value = best_value


class OptunaClusterScanResult(ClusterScanResult):
    """Result of the scan over different clustering hyperparameters."""

    def __init__(self, metrics: dict[str, float], study: optuna.Study):
        super().__init__(
            metrics, best_params=study.best_params, best_value=study.best_value
        )
        self.study = study


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
            return self._scan(start_params=start_params, **kwargs)


class ClusterAlgorithmType(Protocol):
    """Type of a clustering algorithm."""

    def __call__(self, graphs: np.ndarray) -> np.ndarray:
        ...


class ClusterHyperParamScanner(AbstractClusterHyperParamScanner):
    def __init__(
        self,
        *,
        algorithm: ClusterAlgorithmType,
        suggest: Callable[[optuna.trial.Trial], dict[str, Any]],
        data: list[np.ndarray],
        truth: list[np.ndarray],
        pts: list[np.ndarray],
        reconstructable: list[np.ndarray],
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
            data: Data to be clustered
            truth: Truth labels for clustering
            pts: Pt values for each graph
            reconstructable: Whether each hit belongs to a reconstructable true track
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

            def dbscan(data, eps, min_samples):
                return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

            def suggest(trial):
                eps = trial.suggest_float("eps", 1e-5, 1.0)
                min_samples = trial.suggest_int("min_samples", 1, 50)
                return dict(eps=eps, min_samples=min_samples)

            chps = ClusterHyperParamScanner(
                algorithm=dbscan,
                suggest=suggest,
                data=data,
                truth=truths,
                pts=pts,
                guide="v_measure_score",
                metrics=dict(v_measure_score=metrics.v_measure_score),
            )
            study = chps.scan(n_trials=100)
            print(study.best_params)
        """
        self._algorithm = algorithm
        self._suggest = suggest
        assert [len(g) for g in data] == [len(t) for t in truth]
        assert len(data) > 0
        self._data: list[np.ndarray] = data
        self._truth: list[np.ndarray] = truth
        self._pts: list[np.ndarray] = pts
        self._reconstructable: list[np.ndarray] = reconstructable
        self._metrics: dict[str, ClusterMetricType] = metrics
        if sectors is None:
            self._sectors: list[np.ndarray] = [
                np.ones(t, dtype=int) for t in self._truth
            ]
        else:
            assert [len(s) for s in sectors] == [len(t) for t in truth]
            self._sectors = sectors
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
        available: list[int] = np.unique(
            self._sectors[i_graph]
        ).tolist()  # type: ignore
        try:
            available.remove(-1)
        except ValueError:
            pass
        choice = np.random.choice(available).item()
        self._graph_to_sector[i_graph] = choice
        return choice

    def _get_explicit_metric(
        self,
        name: str,
        *,
        predicted: np.ndarray,
        truth: np.ndarray,
        pts: np.ndarray,
        reconstructable: np.ndarray,
    ) -> float:
        """Get metric value from dict of metrics."""
        arguments = dict(
            truth=truth,
            predicted=predicted,
            pts=pts,
            reconstructable=reconstructable,
            pt_thlds=self.pt_thlds,
        )
        if "." in name:
            metric, _, subkey = name.partition(".")
            try:
                return self._metrics[metric](**arguments)[subkey]  # type: ignore
            except KeyError:
                pass
        return self._metrics[name](**arguments)  # type: ignore

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for optuna."""
        params = self._suggest(trial)
        cheap_foms = []
        all_labels = []
        # Do a first run, looking only at the cheap metric, stopping early
        for i_graph, (graph, truth, pts, reconstructable) in enumerate(
            zip(self._data, self._truth, self._pts, self._reconstructable)
        ):
            # Consider a random sector for each graph, but keep the sector consistent
            # between different trials.
            sector = self._get_sector_to_study(i_graph)
            sector_mask = self._sectors[i_graph] == sector
            graph = graph[sector_mask]
            truth = truth[sector_mask]
            pts = pts[sector_mask]
            reconstructable = reconstructable[sector_mask]
            labels = self._algorithm(graph, **params)
            all_labels.append(labels)
            cheap_foms.append(
                self._get_explicit_metric(
                    self._cheap_metric or self._expensive_metric,
                    truth=truth,
                    predicted=labels,
                    pts=pts,
                    reconstructable=reconstructable,
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
            for i_labels, (labels, truth, pts, reconstructable) in enumerate(
                zip(all_labels, self._truth, self._pts, self._reconstructable)
            ):
                sector = self._get_sector_to_study(i_labels)
                sector_mask = self._sectors[i_labels] == sector
                truth = truth[sector_mask]
                expensive_fom = self._get_explicit_metric(
                    self._expensive_metric,
                    truth=truth,
                    predicted=labels,
                    pts=pts,
                    reconstructable=reconstructable,
                )
                expensive_foms.append(expensive_fom)
                if i_labels >= 2:
                    trial.report(
                        np.nanmean(expensive_foms).item(), i_labels + len(self._data)
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

    def _evaluate(
        self, best_params: None | dict[str, float] = None
    ) -> dict[str, float]:
        """Evaluate all metrics (on all sectors and given graphs) for the best
        parameters that we just found with optuna.
        """
        logger.debug("Evaluating all metrics for best clustering")
        with timing("Evaluating all metrics"):
            if best_params is None:
                assert self._study is not None  # mypy
                best_params = self._study.best_params
            return self.__evaluate(best_params=best_params)

    def __evaluate(self, best_params: dict[str, float]) -> dict[str, float]:
        """See _evaluate."""
        metric_values = defaultdict(list)
        for graph, truth, sectors, pts, reconstructable in zip(
            self._data, self._truth, self._sectors, self._pts, self._reconstructable
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
                sector_reconstructable = reconstructable[sector_mask]
                labels = self._algorithm(sector_graph, **best_params)
                for name, metric in self._metrics.items():
                    r = metric(
                        truth=sector_truth,
                        predicted=labels,
                        pts=sector_pts,
                        reconstructable=sector_reconstructable,
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
        if start_params is not None and kwargs.get("n_trials", None) == 1:
            # Do not even start optuna, because that takes time
            logger.debug(
                "Skipping optuna, because start_params are given and only "
                "one trial to run"
            )
            metrics = self._evaluate(best_params=start_params)
            return ClusterScanResult(
                metrics=metrics,
                best_params=start_params,
                best_value=metrics[self._expensive_metric],
            )

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
        return OptunaClusterScanResult(study=self._study, metrics=self._evaluate())
