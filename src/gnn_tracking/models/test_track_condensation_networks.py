from __future__ import annotations

from torch import Tensor as T

from gnn_tracking.models.track_condensation_networks import get_unconnected_node_mask


def test_get_unconnected_node_mask():
    assert (
        get_unconnected_node_mask(
            n_nodes=5,
            edge_index=T(
                [[1, 2], [2, 3]],
            ),
            min_connections=3,
        )
        == T([False, False, False, False, False])
    ).all()

    assert (
        get_unconnected_node_mask(
            n_nodes=5, edge_index=T([[1, 2], [2, 3]]), min_connections=1
        )
        == T([False, True, True, True, False])
    ).all()

    assert (
        get_unconnected_node_mask(
            n_nodes=5, edge_index=T([[1, 2], [2, 3]]), min_connections=2
        )
        == T([False, False, True, False, False])
    ).all()
