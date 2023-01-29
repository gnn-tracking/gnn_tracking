from __future__ import annotations

import copy

import torch
from torch import Tensor as T
from torch_geometric.data import Data


def get_edge_mask_from_node_mask(node_mask: T, edge_index: T) -> T:
    return node_mask[edge_index[0].long()] & node_mask[edge_index[1].long()]


# This will be updated in the next pytorch geometric version.
# Copied directly from the current beta implementation
def mask_select(src: T, dim: int, mask: T) -> T:
    r"""Returns a new tensor which masks the :obj:`src` tensor along the
    dimension :obj:`dim` according to the boolean mask :obj:`mask`.

    Args:
        src (torch.T): The input tensor.
        dim (int): The dimension in which to mask.
        mask (torch.BoolT): The 1-D tensor containing the binary mask to
            index with.
    """
    assert mask.dim() == 1
    assert src.size(dim) == mask.numel()
    dim = dim + src.dim() if dim < 0 else dim
    assert dim >= 0 and dim < src.dim()

    size = [1] * src.dim()
    size[dim] = mask.numel()

    out = src.masked_select(mask.view(size))

    size = list(src.size())
    size[dim] = -1

    return out.view(size)


# This will be updated in the next pytorch geometric version.
# Copied directly from the current beta implementation
def edge_subgraph(data: Data, subset: T) -> Data:
    r"""Returns the induced subgraph given by the edge indices
    :obj:`subset`.
    Will currently preserve all the nodes in the graph, even if they are
    isolated after subgraph computation.

    Args:
        subset (LongT or BoolT): The edges to keep.
    """
    # We need to be very careful here, because we need to preserve the
    # is_edge_attr logic and similar things from the old object
    new_data = copy.copy(data)

    for key, value in data:
        if data.is_edge_attr(key):
            cat_dim = data.__cat_dim__(key, value)
            if subset.dtype == torch.bool:
                new_data[key] = mask_select(value, cat_dim, subset)
            else:
                new_data[key] = value.index_select(cat_dim, subset)

    return new_data
