from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import optuna

from gnn_tracking.metrics.cluster_metrics import ClusterMetricType
from gnn_tracking.utils.earlystopping import no_early_stopping
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.timing import Timer, timing


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

    def get_trial_values(self) -> np.ndarray:
        """Get array with the values of all completed trials."""
        trials_df = self.study.trials_dataframe(attrs=("value", "state")).query(
            "state == 'COMPLETE'"
        )
        return trials_df["value"].to_numpy()


class AbstractClusterHyperParamScanner(ABC):
    """Abstract base class for classes that implement hyperparameter scanning of
    clustering algorithms.
    """

    def __init__(self):
        self.logger = get_logger("ClusterHP")

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
            self.logger.debug("Starting from params: %s", start_params)
        self.logger.info("Starting hyperparameter scan for clustering")
        with timing("Clustering hyperparameter scan & metric evaluation", self.logger):
            return self._scan(start_params=start_params, **kwargs)


class ClusterAlgorithmType(Protocol):
    """Type of a clustering algorithm."""

    def __call__(self, graphs: np.ndarray, **kawrgs) -> np.ndarray:
        ...


def sort_according_to_mask(
    xs: list[np.ndarray], masks: list[np.ndarray] | None = None
) -> list[np.ndarray]:
    """If mask is not `None`, sort vector `x` to first list all elements that are in
    the mask, then all that are masked
    """
    if masks is None:
        masks = [[]] * len(xs)

    def inner(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return x
        else:
            return np.concatenate([x[mask], x[~mask]])

    return [inner(x, mask) for x, mask in zip(xs, masks)]


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
        node_mask: list[np.ndarray] | None = None,
    ):
        """Class to scan hyperparameters of a clustering algorithm.

        Args:
            algorithm: Takes data and keyword arguments
            suggest: Function that suggest parameters to optuna
            data: Data to be clustered
            truth: Truth labels for clustering
            pts: Pt values for each hit
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
            node_mask: If data has been masked before clustering, this is the mask that
                was used. In this case specify the **full** data for `truth`, `pts`,
                `reconstructable` so that the metrics can be calculated taking this
                into account.

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
        super().__init__()
        self._algorithm = algorithm
        self._suggest = suggest
        self._data: list[np.ndarray] = data
        self._truth: list[np.ndarray] = sort_according_to_mask(truth, node_mask)
        self._pts: list[np.ndarray] = sort_according_to_mask(pts, node_mask)
        self._reconstructable: list[np.ndarray] = sort_according_to_mask(
            reconstructable, node_mask
        )
        self._sectors = sort_according_to_mask(sectors, node_mask)
        self._metrics: dict[str, ClusterMetricType] = metrics
        self._es = early_stopping
        self._study = None
        self._cheap_metric = guide_proxy
        self._expensive_metric = guide
        self._graph_to_sector: dict[int, int] = {}
        #: Number of graphs to look at before using accumulated statistics to maybe
        #: prune trial.
        self.pruning_grace_period = 20
        self.pt_thlds = list(pt_thlds)

    @staticmethod
    def _pad_output_with_noise(labels: np.ndarray, length: int) -> np.ndarray:
        """Pad clustering output to length with noise labels."""
        return np.concatenate([labels, np.full(length - len(labels), -1)])

    def _get_sector_to_study(self, i_graph: int) -> int:
        """Return index of sector to study for graph $i_graph."""
        try:
            return self._graph_to_sector[i_graph]
        except KeyError:
            pass
        sectors, counts = np.unique(self._sectors[i_graph], return_counts=True)
        no_noise_mask = sectors >= 0
        chosen_idx = np.argmax(counts[no_noise_mask])
        chosen_sector = sectors[no_noise_mask][chosen_idx]
        return chosen_sector

    # todo: rename
    def _get_explicit_metric(
        self,
        name: str,
        *,
        predicted: np.ndarray,
        truth: np.ndarray,
        pts: np.ndarray,
        reconstructable: np.ndarray,
    ) -> float:
        """Evaluate metric specified by name on clustered data for single graph"""
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

    _EvaluatedMetrics = collections.namedtuple(
        "EvaluatedMetrics",
        ["all_labels", "foms", "clustering_time", "metric_evaluation_time"],
    )

    def _evaluate_metrics(
        self, cluster_params: dict[str, Any], metrics: Iterable[str], all_labels=None
    ) -> ClusterHyperParamScanner._EvaluatedMetrics:
        if all_labels is None:
            all_labels = []
        foms = defaultdict(list)
        clustering_time = 0.0
        metric_evaluation_time = collections.defaultdict(float)
        timer = Timer()
        for i_graph in range(len(self._data)):
            sector = self._get_sector_to_study(i_graph)
            sector_mask = self._sectors[i_graph] == sector
            _data = self._data[i_graph]
            data = _data[sector_mask[: len(_data)]]
            truth = self._truth[i_graph][sector_mask]
            pts = self._pts[i_graph][sector_mask]
            reconstructable = self._reconstructable[i_graph][sector_mask]
            try:
                labels = all_labels[i_graph]
            except IndexError:
                timer()
                labels = self._pad_output_with_noise(
                    self._algorithm(data, **cluster_params), len(reconstructable)
                )
                all_labels.append(labels)
                clustering_time += timer()
            for metric_name in metrics:
                timer()
                r = self._get_explicit_metric(
                    metric_name,
                    truth=truth,
                    predicted=labels,
                    pts=pts,
                    reconstructable=reconstructable,
                )
                if not isinstance(r, Mapping):
                    foms[metric_name].append(r)
                else:
                    for k, v in r.items():
                        foms[f"{metric_name}.{k}"].append(v)
                metric_evaluation_time[metric_name] += timer()
        return self._EvaluatedMetrics(
            all_labels=all_labels,
            foms=foms,
            clustering_time=clustering_time,
            metric_evaluation_time=metric_evaluation_time,
        )

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for optuna."""
        params = self._suggest(trial)
        # Do a first run, looking only at the cheap metric, stopping early
        ems = self._evaluate_metrics(
            params, [self._cheap_metric or self._expensive_metric]
        )
        cheap_foms = ems.foms[self._cheap_metric or self._expensive_metric]
        if not self._cheap_metric:
            # What we just evaluated is actually already the expensive metric
            expensive_foms = cheap_foms
        else:
            ems = self._evaluate_metrics(
                params, [self._expensive_metric], all_labels=ems.all_labels
            )
            expensive_foms = ems.foms[self._expensive_metric]
        global_fom = np.nanmean(expensive_foms).item()
        if self._es(global_fom):
            self.logger.info("Stopped early")
            trial.study.stop()
        return global_fom

    def _evaluate(
        self, best_params: None | dict[str, float] = None
    ) -> dict[str, float]:
        """Evaluate all metrics (on all sectors and given graphs) for the best
        parameters that we just found with optuna.
        """
        self.logger.debug("Evaluating all metrics for best clustering")
        timer = Timer()
        if best_params is None:
            assert self._study is not None  # mypy
            best_params = self._study.best_params
        em = self._evaluate_metrics(best_params, self._metrics.keys())
        metric_timing_str = ", ".join(
            f"{name}: {t}" for name, t in em.metric_evaluation_time.items()
        )
        self.logger.debug(
            "Evaluating metrics took %f seconds: Clustering time: %f, total metric "
            "eval: %f, individual: %s",
            timer(),
            em.clustering_time,
            sum(em.metric_evaluation_time.values()),
            metric_timing_str,
        )
        return {k: np.nanmean(v).item() for k, v in em.foms.items() if v} | {
            f"{k}_std": np.nanstd(v, ddof=1).item() for k, v in em.foms.items() if v
        }

    def _scan(
        self, start_params: dict[str, Any] | None = None, **kwargs
    ) -> ClusterScanResult:
        """Run the scan."""
        self._es.reset()
        if start_params is not None and kwargs.get("n_trials") == 1:
            # Do not even start optuna, because that takes time
            self.logger.debug(
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
        result = OptunaClusterScanResult(study=self._study, metrics=self._evaluate())
        tdf = result.get_trial_values()
        self.logger.debug(
            "Variance among %d trials is %f. Min/max: %f/%f",
            len(tdf),
            tdf.std(ddof=1),
            tdf.min(),
            tdf.max(),
        )
        return result


class ClusterFctType(Protocol):
    """Type of a clustering scanner function"""

    #: Maps the keys from `TCNTrainer.evaluate_model` to the inputs to `__call__`
    required_model_outputs = {
        "x": "graphs",
        "particle_id": "truth",
        "sector": "sectors",
        "pt": "pts",
        "reconstructable": "reconstructable",
        "ec_hit_mask": "node_mask",
    }

    def __call__(
        self,
        graphs: list[np.ndarray],
        truth: list[np.ndarray],
        sectors: list[np.ndarray],
        pts: list[np.ndarray],
        reconstructable: list[np.ndarray],
        epoch=None,
        start_params: dict[str, Any] | None = None,
        node_mask: list[np.ndarray] | None = None,
    ) -> ClusterScanResult:
        ...
