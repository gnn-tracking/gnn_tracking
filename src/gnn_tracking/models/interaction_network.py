from __future__ import annotations

import torch
from models.mlp import MLP
from torch import Tensor
from torch_geometric.nn import MessagePassing


# fixme: Missing abstract methods!
class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        node_indim,
        edge_indim,
        node_outdim=3,
        edge_outdim=4,
        node_hidden_dim=40,
        edge_hidden_dim=40,
        aggr="add",
    ):
        super(InteractionNetwork, self).__init__(aggr=aggr, flow="source_to_target")
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

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return x_tilde, self.E_tilde

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming, x_j --> outgoing
        m = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E_tilde = self.relational_model(m)
        return self.E_tilde

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.object_model(c)
