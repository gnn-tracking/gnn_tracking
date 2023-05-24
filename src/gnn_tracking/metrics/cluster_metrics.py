"""Metrics evaluating the quality of clustering/i.e., the usefulness of the
algorithm for tracking.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import Callable, Iterable, Protocol, TypedDict

import numpy as np
import pandas as pd
from sklearn import metrics

from gnn_tracking.utils.math import zero_division_gives_nan
from gnn_tracking.utils.nomenclature import denote_pt
from gnn_tracking.utils.signature import tolerate_additional_kwargs


class ClusterMetricType(Protocol):
    """Function type that calculates a clustering metric."""

    def __call__(
        self,
        *,
        truth: np.ndarray,
        predicted: np.ndarray,
        pts: np.ndarray,
        reconstructable: np.ndarray,
        pt_thlds: list[float],
    ) -> float | dict[str, float]:
        ...


class TrackingMetrics(TypedDict):
    """Custom cluster metrics for tracking.

    All nominators and denominators only count clusters where the majority particle
    is reconstructable.
    If a pt threshold is applied, the denominator only counts clusters where the
    majority PID's pt is above the threshold.
    """

    #: True number of particles
    n_particles: int
    #: Number of clusters/number of predicted particles. Cleaned means
    n_cleaned_clusters: int
    #: Number of reconstructed tracks (clusters) containing only hits from the same
    #: particle and every hit generated by that particle, divided by the true number
    #: of particles
    perfect: float
    #: The number of reconstructed tracks containing over 50% of hits from the same
    #: particle and over 50% of that particle`s hits, divided by the total number of
    #: true particles
    double_majority: float
    #: The number of reconstructed tracks containing over 75% of hits from the same
    #: particle, divided by the total number reconstructed tracks (clusters)
    lhc: float

    fake_perfect: float
    fake_double_majority: float
    fake_lhc: float


_tracking_metrics_nan_results: TrackingMetrics = {
    "n_particles": 0,
    "n_cleaned_clusters": 0,
    "perfect": float("nan"),
    "lhc": float("nan"),
    "double_majority": float("nan"),
    "fake_perfect": float("nan"),
    "fake_lhc": float("nan"),
    "fake_double_majority": float("nan"),
}


def tracking_metric_df(h_df: pd.DataFrame, predicted_count_thld=3) -> pd.DataFrame:
    """Label clusters as double majority/perfect/LHC.

    Args:
        h_df: Hit information dataframe
        predicted_count_thld: Number of hits a cluster must have to be considered a
            valid cluster

    Returns:
        cluster dataframe with columns such as "double_majority" etc.
    """
    # For each cluster, we determine the true PID that is associated with the most
    # hits in that cluster.
    # Here we make use of the fact that `df.value_counts` sorts by the count.
    # That means that if we group by the cluster and take the first line
    # for each of the counts, we have the most popular PID for each cluster.
    # The resulting dataframe now has both the most popular PID ("id" column) and the
    # number of times it appears ("0" column).
    # This strategy is a significantly (!) faster version than doing
    # c_id.groupby("c").agg(lambda x: x.mode()[0]) etc.
    pid_counts = h_df[["c", "id"]].value_counts().reset_index()

    # Compatibility issue with pandas 2.0 (2.0 uses 'count')
    _count_key = 0 if 0 in pid_counts.columns else "count"

    pid_counts_grouped = pid_counts.groupby("c")
    c_df = pid_counts_grouped.first().rename(
        {"id": "maj_pid", _count_key: "maj_hits"}, axis=1
    )
    # Number of hits per cluster
    c_df["cluster_size"] = pid_counts_grouped[_count_key].sum()
    # Assume that negative cluster labels mean that the cluster was labeled as
    # invalid
    unique_predicted, predicted_counts = np.unique(h_df["c"], return_counts=True)
    # Cluster mask: all clusters that are not labeled as noise and have minimum number
    # of hits.
    c_df["valid_cluster"] = (unique_predicted >= 0) & (
        predicted_counts >= predicted_count_thld
    )

    # Properties associated to PID. This is pretty trivial, but since everything is
    # passed by hit, rather than by PID, we need to get rid of "duplicates"
    particle_properties = list(
        {"pt", "reconstructable", "eta"}.intersection(h_df.columns)
    )
    # Could to .first() for pt/reconstructable, but we want to average over eta
    pid_to_props = (
        h_df[["id"] + particle_properties].groupby("id")[particle_properties].mean()
    )
    c_df = c_df.merge(
        pid_to_props, left_on="maj_pid", right_index=True, copy=False
    ).rename(columns={key: f"maj_{key}" for key in particle_properties})

    # For each PID: Number of hits (in any cluster)
    pid_to_count = Counter(h_df["id"])
    # For each cluster: Take most popular PID of that cluster and get number of hits of
    # that PID (in any cluster)
    c_df["maj_pid_hits"] = c_df["maj_pid"].map(pid_to_count)

    # For each cluster: Fraction of hits that have the most popular PID
    c_df["maj_frac"] = (c_df["maj_hits"] / c_df["cluster_size"]).fillna(0)
    # For each cluster: Take the most popular PID of that cluster. What fraction of
    # the corresponding hits is in this cluster?
    c_df["maj_pid_frac"] = (c_df["maj_hits"] / c_df["maj_pid_hits"]).fillna(0)

    c_df["perfect_match"] = (
        (c_df["maj_pid_hits"] == c_df["maj_hits"])
        & (c_df["maj_frac"] > 0.99)
        & c_df["valid_cluster"]
    )
    c_df["double_majority"] = (
        (c_df["maj_pid_frac"] > 0.5) & (c_df["maj_frac"] > 0.5) & c_df["valid_cluster"]
    )
    c_df["lhc_match"] = (c_df["maj_frac"] > 0.75) & c_df["valid_cluster"]
    return c_df


def count_tracking_metrics(
    c_df: pd.DataFrame, h_df: pd.DataFrame, c_mask: np.ndarray, h_mask: np.ndarray
) -> TrackingMetrics:
    """Calculate TrackingMetrics from cluster and hit information.

    Args:
        c_df: Output dataframe from `tracking_metric_dfs`
        h_df: Hit information dataframe
        c_mask: Cluster mask
        h_mask: Hit mask

    Returns:
        TrackingMetrics namedtuple.
    """
    n_particles = len(np.unique(h_df["id"][h_mask]))
    n_clusters = c_mask.sum()

    n_perfect_match = sum(c_df["perfect_match"][c_mask])
    n_double_majority = sum(c_df["double_majority"][c_mask])
    n_lhc_match = sum(c_df["lhc_match"][c_mask])

    fake_pm = n_clusters - n_perfect_match
    fake_dm = n_clusters - n_double_majority
    fake_lhc = n_clusters - n_lhc_match

    r: TrackingMetrics = {
        "n_particles": n_particles,
        "n_cleaned_clusters": n_clusters,
        "perfect": zero_division_gives_nan(n_perfect_match, n_particles),
        "double_majority": zero_division_gives_nan(n_double_majority, n_particles),
        "lhc": zero_division_gives_nan(n_lhc_match, n_clusters),
        "fake_perfect": zero_division_gives_nan(fake_pm, n_particles),
        "fake_double_majority": zero_division_gives_nan(fake_dm, n_particles),
        "fake_lhc": zero_division_gives_nan(fake_lhc, n_clusters),
    }
    return r


def tracking_metrics(
    *,
    truth: np.ndarray,
    predicted: np.ndarray,
    pts: np.ndarray,
    reconstructable: np.ndarray,
    pt_thlds: Iterable[float],
    predicted_count_thld=3,
) -> dict[float, TrackingMetrics]:
    """Calculate 'custom' metrics for matching tracks and hits.

    Args:
        truth: Truth labels/PIDs for each hit
        predicted: Predicted labels/cluster index for each hit. Negative labels are
            interpreted as noise (because this is how DBSCAN outputs it) and are
            ignored
        pts: pt values of the hits
        reconstructable: Whether the hit belongs to a "reconstructable tracks" (this
            usually implies a cut on the number of layers that are being hit
            etc.)
        pt_thlds: pt thresholds to calculate the metrics for
        predicted_count_thld: Minimal number of hits in a cluster for it to not be
            rejected.

    Returns:
        See `TrackingMetrics`
    """
    for ar in (truth, predicted, pts, reconstructable):
        # Tensors behave differently when counting, so this is absolutely
        # vital!
        assert isinstance(ar, np.ndarray)
    assert predicted.shape == truth.shape == pts.shape, (
        predicted.shape,
        truth.shape,
        pts.shape,
    )
    if len(truth) == 0:
        return {pt: _tracking_metrics_nan_results for pt in pt_thlds}
    h_df = pd.DataFrame(
        {"c": predicted, "id": truth, "pt": pts, "reconstructable": reconstructable}
    )
    c_df = tracking_metric_df(h_df, predicted_count_thld=predicted_count_thld)

    result = dict[float, ClusterMetricType]()
    for pt in pt_thlds:
        c_mask = (
            (c_df["maj_pt"] >= pt) & c_df["maj_reconstructable"] & c_df["valid_cluster"]
        )
        h_mask = (h_df["pt"] >= pt) & h_df["reconstructable"].astype(bool)

        r = count_tracking_metrics(c_df, h_df, c_mask, h_mask)
        result[pt] = r  # type: ignore
    return result  # type: ignore


def flatten_track_metrics(
    custom_metrics_result: dict[float, dict[str, float]]
) -> dict[str, float]:
    """Flatten the result of `custom_metrics` by using pt suffixes to arrive at a
    flat dictionary, rather than a nested one.
    """
    return {
        denote_pt(k, pt): v
        for pt, results in custom_metrics_result.items()
        for k, v in results.items()
    }


def count_hits_per_cluster(predicted: np.ndarray) -> np.ndarray:
    """Count number of hits per cluster"""
    _, counts = np.unique(predicted, return_counts=True)
    hist_counts, _ = np.histogram(counts, bins=np.arange(0.5, counts.max() + 1.5))
    return hist_counts


def hits_per_cluster_count_to_flat_dict(
    counts: np.ndarray, min_max=10
) -> dict[str, float]:
    """Turn result array from `count_hits_per_cluster` into a dictionary
    with cumulative counts.

    Args:
        counts: Result from `count_hits_per_cluster`
        min_max: Pad the counts with zeros to at least this length
    """
    cumulative = np.cumsum(
        np.pad(counts, (0, max(0, min_max - len(counts))), "constant")
    )
    total = cumulative[-1]
    return {
        f"hitcountgeq_{i:04}": cumulative / total
        for i, cumulative in enumerate(reversed(cumulative), start=1)
    }


def _sklearn_signature_wrap(func: Callable) -> ClusterMetricType:
    """A decorator to make an sklearn cluster metric function accept/take the
    arguments from ``ClusterMetricType``.
    """

    @functools.wraps(func)
    @tolerate_additional_kwargs
    def wrapped(predicted: np.ndarray, truth: np.ndarray):
        return func(truth, predicted)

    return wrapped


#: Common metrics that we have for clustering/matching of tracks to hits
common_metrics: dict[str, ClusterMetricType] = {
    "v_measure": _sklearn_signature_wrap(metrics.v_measure_score),
    "homogeneity": _sklearn_signature_wrap(metrics.homogeneity_score),
    "completeness": _sklearn_signature_wrap(metrics.completeness_score),
    "trk": lambda *args, **kwargs: flatten_track_metrics(
        tracking_metrics(*args, **kwargs)
    ),
    "adjusted_rand": _sklearn_signature_wrap(metrics.adjusted_rand_score),
    "fowlkes_mallows": _sklearn_signature_wrap(metrics.fowlkes_mallows_score),
    # adjusted mutual info is very slow
    # "adjusted_mutual_info": _sklearn_signature_wrap(
    # metrics.adjusted_mutual_info_score),
    # "trkc": lambda **kwargs: hits_per_cluster_count_to_flat_dict(
    #     tolerate_additional_kwargs(count_hits_per_cluster)(**kwargs)
    # ),
}
