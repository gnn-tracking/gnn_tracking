from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from models.mlp import MLP
from torch import Tensor
from torch.nn import Linear, ReLU
from torch.nn import Sequential as Seq
from torch.nn import Sigmoid
from torch_geometric.nn import MessagePassing


class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        node_indim,
        edge_indim,
        node_outdim=3,
        edge_outdim=4,
        relational_hidden_size=80,
        object_hidden_size=80,
        aggr="add",
    ):
        super(InteractionNetwork, self).__init__(aggr=aggr, flow="source_to_target")
        self.relational_model = MLP(
            2 * node_indim + edge_indim, edge_outdim, relational_hidden_size
        )
        self.object_model = MLP(
            node_indim + edge_outdim, node_outdim, object_hidden_size
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
