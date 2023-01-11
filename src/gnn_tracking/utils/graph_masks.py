from __future__ import annotations

import numpy as np
import torch
from torch import Tensor as T


def get_edge_mask_from_node_mask(node_mask: T, edge_index: T) -> T:
    return node_mask[edge_index[0].long()] & node_mask[edge_index[1].long()]


def get_edge_index_after_node_mask(
    node_mask: T, edge_index: T, *, edge_mask: T | None = None
) -> tuple[T, T]:
    """A node mask is applied to all nodes.
    All edges that connect to a masked node are removed.
    Because the edge index refers to node
    by their index in the node list, the edge index needs to be updated to
    match this.

    Args:
        edge_index: 2 x n_edges tensor
        node_mask: Mask for all nodes

    Returns:
        edge index (2 x n_edges tensor), edge mask (that is already included in the
        edge index)
    """
    if edge_index.shape[1] == 0:
        # edge mask would also be empty tensor
        return edge_index, T([]).bool().to(edge_index.device)

    implied_edge_mask = get_edge_mask_from_node_mask(node_mask, edge_index)
    if edge_mask is not None:
        implied_edge_mask &= edge_mask
    # Somehow using tensors will mess up the call with np.vectorize
    old_edge_indices = np.arange(len(node_mask))[node_mask.cpu()]
    new_edge_indices = np.arange(node_mask.sum().cpu())
    assert old_edge_indices.shape == new_edge_indices.shape
    edge_index_mapping = np.vectorize(dict(zip(old_edge_indices, new_edge_indices)).get)
    edge_index = torch.stack(
        [
            torch.from_numpy(
                edge_index_mapping(edge_index[0][implied_edge_mask].cpu())
            ),
            torch.from_numpy(
                edge_index_mapping(edge_index[1][implied_edge_mask].cpu())
            ),
        ]
    ).to(edge_index.device)
    return edge_index.long(), implied_edge_mask
