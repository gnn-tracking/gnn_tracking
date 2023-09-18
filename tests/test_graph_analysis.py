from typing import NamedTuple

import networkx as nx
import numpy as np
import pytest
import torch

from gnn_tracking.analysis.graphs import (
    TrackGraphInfo,
    get_all_graph_construction_stats,
    get_cc_labels,
    get_track_graph_info,
)


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
def test_get_all_track_graph_info(test_case: _TestCase):
    assert (
        get_track_graph_info(test_case.graph, np.array(test_case.pids), 0)
        == test_case.tgi
    )


def test_ec_plot_integration(built_graphs):
    _, graph_builder = built_graphs
    g = graph_builder.data_list[0].to("cpu")
    get_all_graph_construction_stats(g)


def test_fixed_graph_construction(built_graphs):
    _, graph_builder = built_graphs
    graphs = graph_builder.data_list

    def _test_for_all(fct, values):
        for graph, value in zip(graphs, values):
            assert fct(graph.cpu()) == pytest.approx(value)

    _test_for_all(lambda g: len(g.x), [843, 750])
    _test_for_all(lambda g: g.x.sum().long(), [15312, -7275])
    _test_for_all(
        lambda g: len(g.y),
        [
            6752,
            5796,
        ],
    )
    _test_for_all(
        lambda g: sum(g.y),
        [
            1860,
            1526,
        ],
    )


def test_get_cc_labels():
    ei = torch.Tensor([[0, 1], [1, 0]])
    assert (get_cc_labels(ei, num_nodes=2) == torch.Tensor([0, 0])).all()
    assert (get_cc_labels(ei, num_nodes=3) == torch.Tensor([0, 0, 1])).all()
    ei = torch.Tensor([[], []])
    assert (get_cc_labels(ei, num_nodes=3) == torch.Tensor([0, 1, 2])).all()
