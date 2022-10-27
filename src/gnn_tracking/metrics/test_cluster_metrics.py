from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pytest import approx

from gnn_tracking.metrics.cluster_metrics import custom_metrics


@dataclasses.dataclass
class ClusterMetricTestCase:
    def __init__(self, *, truth, predicted, **kwargs):
        self.truth = np.array(truth)
        self.predicted = np.array(predicted)
        self.expected = kwargs

    def run(self):
        metrics = custom_metrics(truth=self.truth, predicted=self.predicted)
        assert metrics == approx(self.expected, nan_ok=True)


test_cases = [
    ClusterMetricTestCase(
        truth=[],
        predicted=[],
        total=0,
        perfect=float("nan"),
        lhc=float("nan"),
        double_majority=float("nan"),
    ),
    ClusterMetricTestCase(
        truth=[0],
        predicted=[1],
        n_particles=1,
        n_clusters=1,
        perfect=1.0,
        lhc=1.0,
        double_majority=1.0,
    ),
    ClusterMetricTestCase(
        truth=[0],
        predicted=[0],
        n_particles=1,
        n_clusters=1,
        perfect=1.0,
        lhc=1.0,
        double_majority=1.0,
    ),
    ClusterMetricTestCase(
        truth=[0, 1],
        predicted=[1, 0],
        n_particles=2,
        n_clusters=2,
        perfect=1.0,
        lhc=1.0,
        double_majority=1.0,
    ),
    ClusterMetricTestCase(
        truth=[0, 0],
        predicted=[1, 0],
        n_particles=1,
        n_clusters=2,
        perfect=0.0,
        lhc=2.0 / 2.0,
        double_majority=0.0,
    ),
    ClusterMetricTestCase(
        truth=[1, 0],
        predicted=[0, 0],
        n_particles=2,
        n_clusters=1,
        perfect=0.0,
        lhc=0.0,
        double_majority=0.0,
    ),
    ClusterMetricTestCase(
        truth=[0, 0, 0, 0, 1],
        predicted=[0, 0, 0, 0, 0],
        n_particles=2,
        n_clusters=1,
        perfect=0,
        lhc=1 / 1,
        double_majority=1 / 2,
    ),
    ClusterMetricTestCase(
        truth=[0, 0, 0, 0, 0],
        predicted=[0, 0, 0, 0, 1],
        n_particles=1,
        n_clusters=2,
        perfect=0,
        lhc=2 / 2,
        double_majority=1 / 1,
    ),
    ClusterMetricTestCase(
        # fmt: off
        truth=[
            0, 0, 0, 0, 0, 0,  # lhc, dm
            1, 1, 1, 1, 1, 5,  # lhc
            0, 1, 1, 2,  # x
            0, 1, 2, 3,  # x
            4, 4,  # perfect, lhc, dm
            5  # lhc, dm
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
        n_clusters=6,
        perfect=1 / 6,
        lhc=4 / 6,
        double_majority=3 / 6,
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_custom_metrics(test_case):
    test_case.run()
