from __future__ import annotations

import torch
from torch import Tensor as T
from torch_geometric.nn import MessagePassing

from gnn_tracking.models.mlp import MLP


# noinspection PyAbstractClass
class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        *,
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
        self._e_tilde: T | None = None

    def forward(self, x: T, edge_index: T, edge_attr: T) -> tuple[T, T]:
        """Forward pass

        Args:
            x: Input node features
            edge_index:
            edge_attr: Input edge features

        Returns:
            Output node embedding, output edge embedding
        """
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        assert self._e_tilde is not None  # mypy
        return x_tilde, self._e_tilde

    # noinspection PyMethodOverriding
    def message(self, x_i: T, x_j: T, edge_attr: T) -> T:
        """Calculate message of an edge

        Args:
            x_i: Features of node 1 (node where the edge ends)
            x_j: Features of node 2 (node where the edge starts)
            edge_attr: Edge features

        Returns:
            Message
        """
        m = torch.cat([x_i, x_j, edge_attr], dim=1)
        self._e_tilde = self.relational_model(m)
        assert self._e_tilde is not None  # mypy
        return self._e_tilde

    # noinspection PyMethodOverriding
    def update(self, aggr_out: T, x: T) -> T:
        """Update for node embedding

        Args:
            aggr_out: Aggregated messages of all edges
            x: Node features for the node that receives all edges

        Returns:
            Updated node features/embedding
        """
        c = torch.cat([x, aggr_out], dim=1)
        return self.object_model(c)
