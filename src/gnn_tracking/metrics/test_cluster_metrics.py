import dataclasses

import numpy as np
import pytest
from pytest import approx

from gnn_tracking.metrics.cluster_metrics import (
    count_hits_per_cluster,
    tracking_metrics,
)


@dataclasses.dataclass
class ClusterMetricTestCase:
    def __init__(
        self,
        *,
        truth: list[float],
        predicted: list[float],
        pts: None | list[float] = None,
        reconstructable: None | list[bool] = None,
        pt_thld=-1.0,
        predicted_count_thld=1,
        **kwargs,
    ):
        self.truth = np.array(truth)
        self.predicted = np.array(predicted)
        self.expected = kwargs
        if pts is None:
            self.pts = np.zeros_like(self.predicted)
        else:
            self.pts = np.array(pts)
        if reconstructable is None:
            self.reconstructable = np.full_like(self.predicted, True)
        else:
            self.reconstructable = np.array(reconstructable)
        self.pt_thld = pt_thld
        self.predicted_count_thld = predicted_count_thld

    def run(self):
        metrics = tracking_metrics(
            truth=self.truth,
            predicted=self.predicted,
            pts=self.pts,
            pt_thlds=[self.pt_thld],
            reconstructable=self.reconstructable,
            predicted_count_thld=self.predicted_count_thld,
        )
        assert key_reduction(metrics[self.pt_thld], self.expected) == approx(
            self.expected, nan_ok=True
        )


def key_reduction(dct, keys):
    return {k: v for k, v in dct.items() if k in keys}


test_cases = [
    # Test 0
    ClusterMetricTestCase(
        truth=[],
        predicted=[],
        n_particles=0,
        n_cleaned_clusters=0,
        perfect=float("nan"),
        lhc=float("nan"),
        double_majority=float("nan"),
    ),
    # Test 1
    # Nan because of having only noise from DBSCAN
    ClusterMetricTestCase(
        truth=[1, 2],
        predicted=[-1, -1],
        n_particles=2,
        n_cleaned_clusters=0,
        perfect=0,
        lhc=float("nan"),
        double_majority=0,
    ),
    # Test 2
    ClusterMetricTestCase(
        truth=[0],
        predicted=[0],
        pt_thld=1.0,
        n_particles=0,
        n_cleaned_clusters=0,
        perfect=float("nan"),
        lhc=float("nan"),
        double_majority=float("nan"),
    ),
    # Test 3
    ClusterMetricTestCase(
        truth=[0],
        predicted=[1],
        n_particles=1,
        n_cleaned_clusters=1,
        perfect=1.0,
        lhc=1.0,
        double_majority=1.0,
    ),
    # Test 4
    ClusterMetricTestCase(
        truth=[0, 0, 0, 0],
        predicted=[1, -1, -1, -1],
        n_particles=1,
        n_cleaned_clusters=1,
        perfect=0.0,
        lhc=1.0,
        double_majority=0.0,
    ),
    # Test 5
    ClusterMetricTestCase(
        truth=[0],
        predicted=[0],
        n_particles=1,
        n_cleaned_clusters=1,
        perfect=1.0,
        lhc=1.0,
        double_majority=1.0,
    ),
    # Test 6
    ClusterMetricTestCase(
        truth=[0, 1],
        predicted=[1, 0],
        n_particles=2,
        n_cleaned_clusters=2,
        perfect=1.0,
        lhc=1.0,
        double_majority=1.0,
    ),
    # Test 7
    ClusterMetricTestCase(
        truth=[0, 0],
        predicted=[1, 0],
        n_particles=1,
        n_cleaned_clusters=2,
        perfect=0.0,
        lhc=2.0 / 2.0,
        double_majority=0.0,
    ),
    # Test 8
    ClusterMetricTestCase(
        truth=[1, 0],
        predicted=[0, 0],
        n_particles=2,
        n_cleaned_clusters=1,
        perfect=0.0,
        lhc=0.0,
        double_majority=0.0,
    ),
    # Test 9
    ClusterMetricTestCase(
        truth=[0, 0, 0, 0, 1],
        predicted=[0, 0, 0, 0, 0],
        n_particles=2,
        n_cleaned_clusters=1,
        perfect=0,
        lhc=1 / 1,
        double_majority=1 / 2,
    ),
    # Test 10
    ClusterMetricTestCase(
        truth=[0, 0, 0, 0, 0],
        predicted=[0, 0, 0, 0, 1],
        n_particles=1,
        n_cleaned_clusters=2,
        perfect=0,
        lhc=2 / 2,
        double_majority=1 / 1,
    ),
    # Test 11
    ClusterMetricTestCase(
        # fmt: off
        truth=[
            0, 0, 0, 0, 0, 0,  # lhc, dm
            1, 1, 1, 1, 1, 5,  # lhc, dm
            0, 1, 1, 2,  # x
            0, 1, 2, 3,  # x
            4, 4,  # perfect, lhc, dm
            5  # lhc
        ],
        predicted=[
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4,
            5,
        ],
        # fmt: on
        n_particles=6,
        n_cleaned_clusters=6,
        perfect=1 / 6,
        lhc=4 / 6,
        double_majority=3 / 6,
    ),
    # Test 12
    # same thing as the last, except with pt thresholds masking some particles
    ClusterMetricTestCase(
        # fmt: off
        truth=[
            0, 0, 0, 0, 0, 0,  # lhc, dm  (masked)
            1, 1, 1, 1, 1, 5,  # lhc, dm
            0, 1, 1, 2,  # x
            0, 1, 2, 3,  # x
            4, 4,  # perfect, lhc, dm  (masked)
            5  # lhc
        ],
        # We mask PIDS 0 and 4.
        pts=[
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            0, 1, 1, 1,
            0, 1, 1, 1,
            0, 0,
            1,
        ],
        predicted=[
            0, 0, 0, 0, 0, 0,  # (masked)
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2,
            # (next line masked: this is a bit random, but 0 is taken as most popular
            # PID (out of several options) and that has a pt of 0, so it's masked)
            3, 3, 3, 3,
            4, 4,  # (masked)
            5,
        ],
        # fmt: on
        pt_thld=0.5,
        n_particles=4,
        n_cleaned_clusters=3,
        perfect=0 / 4,
        lhc=2 / 3,
        double_majority=1 / 4,
    ),
    # Test 13
    # same thing as the last, except with pt thresholds and reconstructability
    # masking some particles
    ClusterMetricTestCase(
        # fmt: off
        # particles: 0 (pt masked), 1 (reco masked), 2, 3, 4 (pt masked), 5 ==> Total 3
        truth=[
            0, 0, 0, 0, 0, 0,  # lhc, dm  (pt-masked)
            1, 1, 1, 1, 1, 5,  # lhc, dm (reco-masked)
            0, 1, 1, 2,  # x
            0, 1, 1, 3,  # x
            4, 4,  # perfect, lhc, dm  (pt-masked)
            5  # lhc
        ],
        # We mask PIDS 0 and 4.
        pts=[
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            0, 1, 1, 1,
            0, 1, 1, 1,
            0, 0,
            1,
        ],
        # Everything reconstructable except PID 1
        reconstructable=[
            True, True, True, True, True, True,
            False, False, False, False, False, True,
            True, False, False, True,
            True, False, False, True,
            True, True,
            True,
        ],
        predicted=[
            0, 0, 0, 0, 0, 0,  # (pt-masked)
            1, 1, 1, 1, 1, 1,  # (reco-masked)
            2, 2, 2, 2,  # (reco-masked)
            3, 3, 3, 3,  # (reco-masked)
            4, 4,  # (pt-masked)
            5,
        ],
        # fmt: on
        pt_thld=0.5,
        n_particles=3,
        n_cleaned_clusters=1,
        perfect=0 / 2,
        lhc=1 / 1,
        double_majority=0 / 1,
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_custom_metrics(test_case):
    test_case.run()


def test_count_cluster_hits():
    r = count_hits_per_cluster(np.array([0, 0, 0, 1, 1, 2, 3, 3, 3]))
    assert (r == np.array([1, 1, 2])).all()


def test_fix_cluster_metrics():
    rng = np.random.default_rng(0)
    n_samples = 50
    n_particles = 20
    truth = rng.integers(0, n_particles, size=n_samples)
    predicted = truth + rng.integers(0, 4, size=n_samples)
    pt_thlds = [
        0,
        0.5,
        0.9,
    ]
    pts = rng.uniform(0, 3, size=n_samples)[truth]
    reconstructable = rng.choice([True, False], size=n_particles)[truth]
    print(truth.sum(), predicted.sum(), pts.sum(), reconstructable.sum())
    r = tracking_metrics(
        truth=truth,
        predicted=predicted,
        pts=pts,
        reconstructable=reconstructable,
        pt_thlds=pt_thlds,
        predicted_count_thld=3,
    )
    expected = {
        0: {
            "n_particles": 10,
            "n_cleaned_clusters": 4,
            "perfect": 0.0,
            "double_majority": 0.1,
            "lhc": 0.0,
            "fake_perfect": 0.4,
            "fake_double_majority": 0.3,
            "fake_lhc": 1.0,
        },
        0.5: {
            "n_particles": 8,
            "n_cleaned_clusters": 4,
            "perfect": 0.0,
            "double_majority": 0.125,
            "lhc": 0.0,
            "fake_perfect": 0.5,
            "fake_double_majority": 0.375,
            "fake_lhc": 1.0,
        },
        0.9: {
            "n_particles": 6,
            "n_cleaned_clusters": 3,
            "perfect": 0.0,
            "double_majority": 0.16666666666666666,
            "lhc": 0.0,
            "fake_perfect": 0.5,
            "fake_double_majority": 0.3333333333333333,
            "fake_lhc": 1.0,
        },
    }
    for thld in pt_thlds:
        assert r[thld] == approx(expected[thld])
