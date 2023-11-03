from torch import Tensor as T
from torch_geometric.data import Data


def get_good_node_mask(data: Data, *, pt_thld: float = 0.9, max_eta: float = 4.0) -> T:
    """Get a mask for nodes that are included in metrics and more.
    This includes lower limit on pt, not noise, reconstructable, cut on eta.
    """
    return get_good_node_mask_tensors(
        pt=data.pt,
        particle_id=data.particle_id,
        reconstructable=data.reconstructable,
        eta=data.eta,
        pt_thld=pt_thld,
        max_eta=max_eta,
    )


def get_good_node_mask_tensors(
    *, pt, particle_id, reconstructable, eta, pt_thld: float = 0.9, max_eta: float = 4.0
) -> T:
    """See `get_good_node_mask`"""
    return (
        (pt > pt_thld)
        & (particle_id > 0)
        & (reconstructable > 0)
        & (eta.abs() < max_eta)
    )


def get_edge_mask_from_node_mask(node_mask: T, edge_index: T) -> T:
    """Get a mask for edges that are between two nodes that are both in the node
    mask.
    """
    return node_mask[edge_index[0].long()] & node_mask[edge_index[1].long()]
