from __future__ import annotations

import numpy as np
import pandas as pd


def get_effs(c, h, particle_id):
    c_id = pd.DataFrame({"c": c, "id": particle_id})
    clusters = c_id.groupby("c")
    majority_counts = clusters["id"].apply(lambda x: sum(x == x.mode()[0]))
    majority_fraction = clusters["id"].apply(lambda x: sum(x == x.mode()[0]) / len(x))
    h_id = pd.DataFrame({"hits": np.ones(len(h)), "id": particle_id})
    particles = h_id.groupby("id")
    nhits = particles["hits"].apply(lambda x: len(x)).to_dict()
    majority_hits = clusters["id"].apply(lambda x: x.mode().map(nhits)[0])
    perfect_match = (majority_hits == majority_counts) & (majority_fraction > 0.99)
    double_majority = ((majority_counts / majority_hits).fillna(0) > 0.5) & (
        majority_fraction > 0.5
    )
    lhc_match = (majority_fraction).fillna(0) > 0.75
    return {
        "total": len(np.unique(c)),
        "perfect": sum(perfect_match),
        "double_majority": sum(double_majority),
        "lhc": sum(lhc_match),
    }
