from __future__ import annotations

import pytest


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
