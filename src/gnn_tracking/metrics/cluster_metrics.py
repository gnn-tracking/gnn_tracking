from __future__ import annotations

from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn import metrics

#: Function type that calculates a clustering metric. The truth labels must be given
#: as first parameter, the predicted labels as second parameter.
metric_type = Callable[[np.ndarray, np.ndarray], Union[float, dict[str, float]]]


def custom_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate 'custom' metrics for matching tracks and hits.

    Args:
        truth: Trut labels/PIDs
        predicted: Predicted labels/PIDs

    Returns:

    """
    assert predicted.shape == truth.shape
    if len(truth) == 0:
        return {
            "total": 0,
            "perfect": float("nan"),
            "lhc": float("nan"),
            "double_majority": float("nan"),
        }
    c_id = pd.DataFrame({"c": predicted, "id": truth})
    clusters = c_id.groupby("c")
    # For each cluster: Take most popular PID of that cluster and get number of
    # corresponding hits in that cluster
    in_cluster_maj_hits = clusters["id"].apply(lambda x: sum(x == x.mode()[0]))
    # For each cluster: Fraction of hits that have the most popular PID
    in_cluster_maj_frac = clusters["id"].apply(lambda x: sum(x == x.mode()[0]) / len(x))
    h_id = pd.DataFrame({"hits": np.ones(len(predicted)), "id": truth})
    particles = h_id.groupby("id")
    # For each PID: Number of hits
    pid_to_count = particles["hits"].apply(lambda x: len(x)).to_dict()
    # For each cluster: Take most popular PID of that cluster and get number of hits of
    # that PID (in any cluster)
    majority_hits = clusters["id"].apply(lambda x: x.mode().map(pid_to_count)[0])
    perfect_match = (majority_hits == in_cluster_maj_hits) & (
        in_cluster_maj_frac > 0.99
    )
    double_majority = ((in_cluster_maj_hits / majority_hits).fillna(0) > 0.5) & (
        in_cluster_maj_frac > 0.5
    )
    lhc_match = in_cluster_maj_frac.fillna(0) > 0.75
    n_particles = len(np.unique(truth))
    n_clusters = len(np.unique(predicted))
    return {
        "n_particles": n_particles,
        "n_clusters": n_clusters,
        "perfect": sum(perfect_match) / n_particles,
        "double_majority": sum(double_majority) / n_particles,
        "lhc": sum(lhc_match) / n_clusters,
    }


#: Common metrics that we have for clustering/matching of tracks to hits
common_metrics: dict[str, metric_type] = {
    "v_measure": metrics.v_measure_score,
    "homogeneity": metrics.homogeneity_score,
    "completeness": metrics.completeness_score,
    "trk": custom_metrics,
    "adjusted_rand": metrics.adjusted_rand_score,
    "fowlkes_mallows": metrics.fowlkes_mallows_score,
    "adjusted_mutual_info": metrics.adjusted_mutual_info_score,
}
