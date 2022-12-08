from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing

from gnn_tracking.models.mlp import MLP


class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        node_indim: int,
        edge_indim: int,
        node_outdim=3,
        edge_outdim=4,
        node_hidden_dim=40,
        edge_hidden_dim=40,
        aggr="add",
    ):
        """Interaction Network, consisting of a relational model and an object model,
        both represented as MLPs.

        Args:
            node_indim: Node feature dimension
            edge_indim: Edge feature dimension
            node_outdim: Output node feature dimension
            edge_outdim: Output edge feature dimension
            node_hidden_dim: Hidden dimension for the object model MLP
            edge_hidden_dim: Hidden dimension for the relational model MLP
            aggr: How to aggregate the messages
        """
        super().__init__(aggr=aggr, flow="source_to_target")
        self.relational_model = MLP(
            2 * node_indim + edge_indim,
            edge_outdim,
            edge_hidden_dim,
        )
        self.object_model = MLP(
            node_indim + edge_outdim,
            node_outdim,
            node_hidden_dim,
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor]:
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return x_tilde, self.E_tilde

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming, x_j --> outgoing
        m = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E_tilde = self.relational_model(m)
        return self.E_tilde

    def update(self, aggr_out, x):
        # aggr_output: Output of aggregating all messages
        # x: Input node features
        c = torch.cat([x, aggr_out], dim=1)
        return self.object_model(c)
