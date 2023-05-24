from __future__ import annotations

from gnn_tracking.analysis.edge_classification import (
    ThresholdTrackInfoPlot,
    collect_all_ec_stats,
)
from gnn_tracking.models.edge_classifier import ECForGraphTCN


def test_ec_plot_integration(built_graphs):
    _, graph_builder = built_graphs
    g = graph_builder.data_list[0]
    node_indim = g.x.shape[1]
    edge_indim = g.edge_attr.shape[1]
    ec = ECForGraphTCN(
        node_indim=node_indim,
        edge_indim=edge_indim,
        hidden_dim=2,
        L_ec=2,
    )
    df = collect_all_ec_stats(ec, [g], thresholds=[0, 0.5, 1.0])  # type: ignore
    ThresholdTrackInfoPlot(df).plot()
