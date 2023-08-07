from torch_geometric.data import Data


def get_efficiency_purity_edges(data: Data, pt_thld=0.9) -> dict[str, float]:
    """Calculate efficiency and purity for edges based on ``data.true_edge_index``.

    Only edges where at least one of the two nodes is accepted by the pt threshold
    (and reconstructable etc.) are considered.
    """
    hit_mask = (data.pt > pt_thld) & (data.particle_id > 0) & (data.reconstructable > 0)
    edge_mask = hit_mask[data.edge_index[0]] | hit_mask[data.edge_index[1]]
    true_edge_mask = (
        hit_mask[data.true_edge_index[0]] & hit_mask[data.true_edge_index[1]]
    )
    # Factor of 2 because the true edges are undirected
    efficiency = data.y[edge_mask].sum() / (2 * true_edge_mask.sum())
    purity = data.y[edge_mask].sum() / edge_mask.sum()
    return {
        "efficiency": efficiency.item(),
        "purity": purity.item(),
    }
