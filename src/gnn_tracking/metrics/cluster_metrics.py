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
    majority_counts = clusters["id"].apply(lambda x: sum(x == x.mode()[0]))
    majority_fraction = clusters["id"].apply(lambda x: sum(x == x.mode()[0]) / len(x))
    h_id = pd.DataFrame({"hits": np.ones(len(predicted)), "id": truth})
    particles = h_id.groupby("id")
    nhits = particles["hits"].apply(lambda x: len(x)).to_dict()
    majority_hits = clusters["id"].apply(lambda x: x.mode().map(nhits)[0])
    perfect_match = (majority_hits == majority_counts) & (majority_fraction > 0.99)
    double_majority = ((majority_counts / majority_hits).fillna(0) > 0.5) & (
        majority_fraction > 0.5
    )
    lhc_match = majority_fraction.fillna(0) > 0.75
    total = len(np.unique(truth))
    return {
        "total": total,
        "perfect": sum(perfect_match) / total,
        "double_majority": sum(double_majority) / total,
        "lhc": sum(lhc_match) / total,
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
