import collections
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import optuna
from pytorch_lightning.core.mixins import HyperparametersMixin

from gnn_tracking.metrics.cluster_metrics import ClusterMetricType
from gnn_tracking.utils.earlystopping import no_early_stopping
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.timing import Timer


class ClusterScanResult:
    """Result of scan over different clustering hyperparameters."""

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


def get_majority_sector(sectors: np.ndarray) -> int:
    """Return most frequent sector that is not noise."""
    unique_sectors, counts = np.unique(sectors, return_counts=True)
    no_noise_mask = unique_sectors >= 0
    if not np.any(no_noise_mask):
        msg = "Only noise in this graph"
        raise ValueError(msg)
    chosen_idx = np.argmax(counts[no_noise_mask])
    return unique_sectors[no_noise_mask][chosen_idx]


# todo: Could simplify this implementation if we pass around DataFrames rather than
#   lots of numpy arrays
class ClusterHyperParamScanner:
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
                returns a dict, the key should be `key.subkey`.
            metrics: Dictionary of metrics to evaluate. Each metric is a function that
                takes truth and predicted labels as numpy arrays and returns a float.
            sectors: List of 1D arrays of sector indices (answering which sector each
                hit from each graph belongs to). If None, all hits are assumed to be
                from the same sector.
            early_stopping: Callable that can be called with result and has a reset
                method. If it returns True, the scan is stopped.
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
        self.logger = get_logger("ClusterHP")
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
        self._guide = guide
        #: Cache for sector to study for each graph.
        self._graph_to_sector = [
            get_majority_sector(sectors) for sectors in self._sectors
        ]
        #: Number of graphs to look at before using accumulated statistics to maybe
        #: prune trial.
        self.pt_thlds = list(pt_thlds)

    @staticmethod
    def _pad_output_with_noise(labels: np.ndarray, length: int) -> np.ndarray:
        """Pad clustering output to length with noise labels."""
        return np.concatenate([labels, np.full(length - len(labels), -1)])

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
        arguments = {
            "truth": truth,
            "predicted": predicted,
            "pts": pts,
            "reconstructable": reconstructable,
            "pt_thlds": self.pt_thlds,
        }
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
    ) -> "ClusterHyperParamScanner._EvaluatedMetrics":
        if all_labels is None:
            all_labels = []
        foms = defaultdict(list)
        clustering_time = 0.0
        metric_evaluation_time = collections.defaultdict(float)
        timer = Timer()
        for i_graph in range(len(self._data)):
            sector = self._graph_to_sector[i_graph]
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
        ems = self._evaluate_metrics(params, [self._guide])
        global_fom = np.nanmean(ems.foms[self._guide]).item()
        self.logger.debug("Evaluated %s: %s", params, global_fom)
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

    def scan(
        self, start_params: dict[str, Any] | None = None, **kwargs
    ) -> ClusterScanResult:
        if kwargs.get("n_trials") == 0:
            self.logger.debug("No cluster scan because n_trials=0")
            return ClusterScanResult(
                metrics={},
                best_params={self._guide: np.nan},
                best_value=np.nan,
            )
        if start_params is not None:
            self.logger.debug("Starting from params: %s", start_params)

        self.logger.info("Starting hyperparameter scan for clustering")
        timer = Timer()
        self._es.reset()
        if start_params is not None and kwargs.get("n_trials") == 1:
            self.logger.debug(
                "Skipping optuna, because start_params are given and only "
                "one trial to run"
            )
            metrics = self._evaluate(best_params=start_params)
            self.logger.info(
                "Clustering hyperparameter scan & metric evaluation took %s", timer()
            )
            return ClusterScanResult(
                metrics=metrics,
                best_params=start_params,
                best_value=metrics[self._guide],
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

        self.logger.info(
            "Clustering hyperparameter scan & metric evaluation took %s", timer()
        )
        return result


class ClusterFctType(Protocol):
    """Type of a clustering scanner function"""

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


class PulsedNTrials(HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        warmup_epoch: int = 0,
        low_trials: int,
        high_trials: int,
        high_every: int = 2,
        warmup_trials: int | None = None,
    ):
        """A parameterization of a simple scheme to set the number of trials.
        Because cluster scans are expensive, you might want to only rescan every
        couple of epochs (and start from the previous best parameter otherwise).

        Args:
            warmup_epoch: If `epoch < warmup_epoch`, use `warmup_trials` trials.
            low_trials: Low number of trials
            high_trials: High number of trials
            every_epoch: If `epoch % every_epoch == 0`, use `high_trials`, else
                `low_trials`.
            warmup_trials: Trials during warmup phase. If None, use low_trials.
        """
        super().__init__()
        self.save_hyperparameters()

    def __call__(self, epoch: int) -> int:
        if epoch < self.hparams.warmup_epoch:
            if self.hparams.warmup_trials is None:
                return self.hparams.low_trials
            return self.hparams.warmup_trials
        if epoch % self.hparams.high_every == 0:
            return self.hparams.high_trials
        return self.hparams.low_trials
