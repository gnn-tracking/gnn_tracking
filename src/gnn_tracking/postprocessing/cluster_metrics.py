from __future__ import annotations

import numpy as np
import pandas as pd


def custom_metrics(labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Custom metrics

    Args:
        labels: Predicted labels
        truth:

    Returns:

    """
    assert labels.shape == truth.shape
    c_id = pd.DataFrame({"c": labels, "id": truth})
    clusters = c_id.groupby("c")
    majority_counts = clusters["id"].apply(lambda x: sum(x == x.mode()[0]))
    majority_fraction = clusters["id"].apply(lambda x: sum(x == x.mode()[0]) / len(x))
    h_id = pd.DataFrame({"hits": np.ones(len(labels)), "id": truth})
    particles = h_id.groupby("id")
    nhits = particles["hits"].apply(lambda x: len(x)).to_dict()
    majority_hits = clusters["id"].apply(lambda x: x.mode().map(nhits)[0])
    perfect_match = (majority_hits == majority_counts) & (majority_fraction > 0.99)
    double_majority = ((majority_counts / majority_hits).fillna(0) > 0.5) & (
        majority_fraction > 0.5
    )
    lhc_match = (majority_fraction).fillna(0) > 0.75
    total = len(np.unique(labels))
    return {
        "total": total,
        "perfect": sum(perfect_match) / total,
        "double_majority": sum(double_majority) / total,
        "lhc": sum(lhc_match) / total,
    }
