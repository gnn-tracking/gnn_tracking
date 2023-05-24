from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import numpy as np
import pytest

from gnn_tracking.analysis.graphs import TrackGraphInfo, get_track_graph_info


class _TestCase(NamedTuple):
    graph: nx.Graph
    pids: list[int]
    tgi: TrackGraphInfo


_test_cases = [
    _TestCase(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)]),
        pids=[0, 0, 0, 0, 0],
        tgi=TrackGraphInfo(
            pid=0,
            n_hits=5,
            n_segments=1,
            n_hits_largest_segment=5,
            distance_largest_segments=0,
            n_hits_largest_component=5,
        ),
    ),
    _TestCase(
        graph=nx.Graph([(0, 1), (2, 3), (3, 4)]),
        pids=[0, 0, 0, 0, 0],
        tgi=TrackGraphInfo(
            pid=0,
            n_hits=5,
            n_segments=2,
            n_hits_largest_segment=3,
            distance_largest_segments=np.inf,
            n_hits_largest_component=3,
        ),
    ),
    _TestCase(
        graph=nx.Graph([(0, 1), (2, 3), (3, 4), (1, 10), (10, 2)]),
        pids=[0, 0, 0, 0, 0],
        tgi=TrackGraphInfo(
            pid=0,
            n_hits=5,
            n_segments=2,
            n_hits_largest_segment=3,
            distance_largest_segments=2,
            n_hits_largest_component=5,
        ),
    ),
]


@pytest.mark.parametrize("test_case", _test_cases)
def test_get_track_graph_info(test_case: _TestCase):
    assert (
        get_track_graph_info(test_case.graph, np.array(test_case.pids), 0)
        == test_case.tgi
    )
