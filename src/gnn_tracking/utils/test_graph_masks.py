from __future__ import annotations

import dataclasses

import pytest
from torch import Tensor as T

from gnn_tracking.utils.graph_masks import get_edge_index_after_node_mask


@dataclasses.dataclass
class GetEdgeIndexAfterNodeMaskTestCase:
    edge_index: T
    node_mask: T
    new_edge_index: T
    edge_mask: T


test_cases = [
    GetEdgeIndexAfterNodeMaskTestCase(
        edge_index=T([[0, 0, 1, 1, 2], [1, 2, 3, 2, 3]]),
        node_mask=T([True, False, True, False]).bool(),
        new_edge_index=T([[0], [1]]),
        edge_mask=T([False, True, False, False, False]),
    ),
    GetEdgeIndexAfterNodeMaskTestCase(
        edge_index=T([[0, 0, 1, 1, 2], [1, 2, 3, 2, 3]]),
        node_mask=T([True, True, True, True]).bool(),
        new_edge_index=T([[0, 0, 1, 1, 2], [1, 2, 3, 2, 3]]),
        edge_mask=T([True, True, True, True, True]),
    ),
    GetEdgeIndexAfterNodeMaskTestCase(
        edge_index=T([[0, 0, 1, 1, 2], [1, 2, 3, 2, 3]]),
        node_mask=T([False, True, True, True]).bool(),
        new_edge_index=T([[0, 0, 1], [2, 1, 2]]),
        edge_mask=T([False, False, True, True, True]),
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_get_edge_index_after_node_mask(test_case):
    new_edge_index, edge_mask = get_edge_index_after_node_mask(
        test_case.node_mask, test_case.edge_index
    )
    assert (new_edge_index == test_case.new_edge_index).all()
    assert (edge_mask == test_case.edge_mask).all()
