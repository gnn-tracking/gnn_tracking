from __future__ import annotations

from collections import Counter
from typing import Callable, TypedDict, Union

import numpy as np
import pandas as pd
from sklearn import metrics

#: Function type that calculates a clustering metric. The truth labels must be given
#: as first parameter, the predicted labels as second parameter.
metric_type = Callable[[np.ndarray, np.ndarray], Union[float, dict[str, float]]]


class CustomMetrics(TypedDict):
    n_particles: int
    n_clusters: int
    perfect: float
    double_majority: float
    lhc: float


def custom_metrics(truth: np.ndarray, predicted: np.ndarray) -> CustomMetrics:
    """Calculate 'custom' metrics for matching tracks and hits.

    Args:
        truth: Truth labels/PIDs
        predicted: Predicted labels/PIDs

    Returns:

    """
    assert predicted.shape == truth.shape
    if len(truth) == 0:
        r: CustomMetrics = {
            "n_particles": 0,
            "n_clusters": 0,
            "perfect": float("nan"),
            "lhc": float("nan"),
            "double_majority": float("nan"),
        }
        return r
    c_id = pd.DataFrame({"c": predicted, "id": truth})
    # Here we make use of the fact that value_counts sorts by the count
    # So after we have the count, we group by the cluster and take the first line
    # for each
    pid_counts = c_id.value_counts().reset_index()
    majority_df = pid_counts.groupby("c").first()
    # This dataframe now has both the most popular PID ("id" column) and the
    # number of times it appears ("0" column)
    in_cluster_maj_pids = majority_df["id"]
    in_cluster_maj_hits = majority_df[0]
    # This is a significantly (!) faster version than doing
    # c_id.groupby("c").agg(lambda x: x.mode()[0]) etc.
    cluster_sizes = pid_counts.groupby("c")[0].sum()
    # For each cluster: Fraction of hits that have the most popular PID
    in_cluster_maj_frac = (in_cluster_maj_hits / cluster_sizes).fillna(0)
    # For each PID: Number of hits
    pid_to_count = Counter(truth)
    # For each cluster: Take most popular PID of that cluster and get number of hits of
    # that PID (in any cluster)
    majority_hits = in_cluster_maj_pids.map(pid_to_count)
    perfect_match = (majority_hits == in_cluster_maj_hits) & (
        in_cluster_maj_frac > 0.99
    )
    double_majority = ((in_cluster_maj_hits / majority_hits).fillna(0) > 0.5) & (
        in_cluster_maj_frac > 0.5
    )
    lhc_match = in_cluster_maj_frac > 0.75
    n_particles = len(np.unique(truth))
    n_clusters = len(np.unique(predicted))
    r = {
        "n_particles": n_particles,
        "n_clusters": n_clusters,
        "perfect": sum(perfect_match) / n_particles,
        "double_majority": sum(double_majority) / n_particles,
        "lhc": sum(lhc_match) / n_clusters,
    }
    return r


#: Common metrics that we have for clustering/matching of tracks to hits
common_metrics: dict[str, metric_type] = {
    "v_measure": metrics.v_measure_score,
    "homogeneity": metrics.homogeneity_score,
    "completeness": metrics.completeness_score,
    "trk": custom_metrics,  # type: ignore
    "adjusted_rand": metrics.adjusted_rand_score,
    "fowlkes_mallows": metrics.fowlkes_mallows_score,
    "adjusted_mutual_info": metrics.adjusted_mutual_info_score,
}
