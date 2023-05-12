from __future__ import annotations

import pytest


def test_fixed_graph_construction(built_graphs):
    _, graph_builder = built_graphs
    graphs = graph_builder.data_list

    def _test_for_all(fct, values):
        for graph, value in zip(graphs, values):
            assert fct(graph) == pytest.approx(value)

    _test_for_all(lambda g: len(g.x), [750, 843])
    _test_for_all(lambda g: g.x.sum().long(), [-9555, 12274])
    _test_for_all(lambda g: len(g.y), [5796, 6752])
    _test_for_all(lambda g: sum(g.y), [1526, 1860])
