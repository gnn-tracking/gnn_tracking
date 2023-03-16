from __future__ import annotations

import torch
from torch import Tensor
from torch import Tensor as T
from torch import nn
from torch.nn.functional import relu

from gnn_tracking.models.interaction_network import InteractionNetwork


@torch.jit.script
def convex_combination(
    *,
    delta: T,
    residue: T,
    alpha_residue: float,
) -> T:
    """Convex combination of delta and residue"""
    # if torch.isclose(alpha_residue, 0.0):
    #     return relu(delta)
    assert 0 <= alpha_residue <= 1
    return alpha_residue * residue + (1 - alpha_residue) * delta


class ResIN(nn.Module):
    def __init__(
        self,
        layers: list[nn.Module],
        *,
        node_dim: int,
        edge_dim: int,
        alpha_node: float = 0.5,
        alpha_edge: float = 0.5,
        add_bn: bool = True,
    ):
        """Apply a list of layers in sequence with residual connections for the nodes.
        Built for interaction networks, but any network that returns a node feature
        tensor and an edge feature tensor should
        work.

        Note that a ReLu activation function is applied to the node result of the
        layer.

        Args:
            layers: List of layers
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            alpha_node: Strength of the node embedding residual connection
            alpha_edge: Strength of the edge embedding residual connection
            add_bn: Add batch norms
        """
        super().__init__()
        if not len(layers) % 2 == 0:
            raise ValueError("Only even number of layers allowed at the moment")
        self._layers = nn.ModuleList(layers)
        if add_bn:
            self._node_batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(node_dim) for _ in range(len(layers))]
            )
            self._edge_batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(edge_dim) for _ in range(len(layers))]
            )
        else:
            self._node_batch_norms = nn.ModuleList(
                [nn.Identity() for _ in range(len(layers))]
            )
            self._edge_batch_norms = nn.ModuleList(
                [nn.Identity() for _ in range(len(layers))]
            )

        self._alpha_node = alpha_node
        self._alpha_edge = alpha_edge

    @classmethod
    def identical_in_layers(
        cls,
        *,
        node_dim: int,
        edge_dim: int,
        object_hidden_dim=40,
        relational_hidden_dim=40,
        alpha_node: float = 0.5,
        alpha_edge: float = 0.5,
        n_layers=1,
        **kwargs,
    ) -> ResIN:
        """Create a ResIN with identical layers of interaction networks except for
        the first and last one (different dimensions)

        If the input/hidden/output dimensions for the nodes are not the same, MLPs are
        used to map the previous output for the residual connection.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            object_hidden_dim: Hidden dimension for the object model MLP
            relational_hidden_dim: Hidden dimension for the relational model MLP
            alpha_node: Strength of the node residual connection
            alpha_edge: Strength of the edge residual connection
            n_layers: Total number of layers
            **kwargs: Passed on to __init__
        """
        layers = [
            InteractionNetwork(
                node_indim=node_dim,
                edge_indim=edge_dim,
                node_outdim=node_dim,
                edge_outdim=edge_dim,
                node_hidden_dim=object_hidden_dim,
                edge_hidden_dim=relational_hidden_dim,
            )
            for i in range(n_layers)
        ]
        mod = cls(
            layers,
            alpha_node=alpha_node,
            alpha_edge=alpha_edge,
            node_dim=node_dim,
            edge_dim=edge_dim,
            **kwargs,
        )
        return mod

    def forward(self, x, edge_index, edge_attr) -> tuple[Tensor, Tensor]:
        """Forward pass

        Args:
            x: Node features
            edge_index:
            edge_attr: Edge features

        Returns:
            node embedding, edge_embedding
        """
        for i_layer_pair in range(len(self._layers) // 2):
            i0 = 2 * i_layer_pair
            hidden_x, hidden_edge_attr = self._layers[i0](
                relu(self._node_batch_norms[i0](x)),
                edge_index,
                relu(self._edge_batch_norms[i0](edge_attr)),
            )
            i1 = 2 * i_layer_pair + 1
            delta_x, delta_edge_attr = self._layers[i1](
                relu(self._node_batch_norms[i1](hidden_x)),
                edge_index,
                relu(self._edge_batch_norms[i1](hidden_edge_attr)),
            )
            x = convex_combination(
                delta=delta_x, residue=x, alpha_residue=self._alpha_node
            )
            edge_attr = convex_combination(
                delta=delta_edge_attr, residue=edge_attr, alpha_residue=self._alpha_edge
            )
        return x, edge_attr
