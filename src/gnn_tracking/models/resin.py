from __future__ import annotations

from torch import Tensor, nn
from torch.nn.functional import relu

from gnn_tracking.models.interaction_network import InteractionNetwork


class ResIN(nn.Module):
    def __init__(self, layers: list[nn.Module], alpha: float = 0.5):
        """Apply a list of layers in sequence. Built for interaction networks, but any
        network that returns a node feature tensor and an edge feature tensor should
        work.

        Note that a ReLu activation function is applied to the node result of the
        layer.

        Args:
            layers: List of layers
            alpha: Strength of the residual connection
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.alpha = alpha

    @classmethod
    def identical_in_layers(
        cls,
        *,
        node_indim: int,
        edge_indim: int,
        node_hidden_dim: int,
        edge_hidden_dim: int,
        node_outdim=3,
        edge_outdim=4,
        object_hidden_dim=40,
        relational_hidden_dim=40,
        alpha: float = 0.5,
        n_layers=1,
    ) -> ResIN:
        """Create a ResIN with identical layers of interaction networks except for
        the first and last one (different dimensions)

        Args:
            node_indim: Node feature dimension
            edge_indim: Edge feature dimension
            node_hidden_dim: Node feature dimension for the hidden layers
            edge_hidden_dim: Edge feature dimension for the hidden layers
            node_outdim: Output node feature dimension
            edge_outdim: Output edge feature dimension
            object_hidden_dim: Hidden dimension for the object model MLP
            relational_hidden_dim: Hidden dimension for the relational model MLP
            alpha: Strength of the residual connection
            n_layers: Total number of layers
        """
        first_layer = InteractionNetwork(
            node_indim=node_indim,
            edge_indim=edge_indim,
            node_outdim=node_hidden_dim,
            edge_outdim=edge_hidden_dim,
            node_hidden_dim=object_hidden_dim,
            edge_hidden_dim=relational_hidden_dim,
        )
        hidden_layers = [
            InteractionNetwork(
                node_indim=node_hidden_dim,
                edge_indim=edge_hidden_dim,
                node_outdim=node_hidden_dim,
                edge_outdim=edge_hidden_dim,
                node_hidden_dim=object_hidden_dim,
                edge_hidden_dim=relational_hidden_dim,
            )
            for _ in range(n_layers - 2)
        ]
        last_layer = InteractionNetwork(
            node_indim=node_hidden_dim,
            edge_indim=edge_hidden_dim,
            node_outdim=node_outdim,
            edge_outdim=edge_outdim,
            node_hidden_dim=object_hidden_dim,
            edge_hidden_dim=relational_hidden_dim,
        )
        layers = [first_layer, *hidden_layers, last_layer]
        assert len(layers) == n_layers
        return cls(layers, alpha)

    def forward(
        self, h, edge_index, edge_attr
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """

        Args:
            h: Node features
            edge_index:
            edge_attr: Edge features

        Returns:
            Last node features, node features at each layer (including the input),
            edge features at each layer (including the input)
        """
        edge_attrs = [edge_attr]
        hs = [h]
        for layer in self.layers:
            delta_h, edge_attr = layer(h, edge_index, edge_attr)
            h = self.alpha * h + (1 - self.alpha) * relu(delta_h)
            hs.append(h)
            edge_attrs.append(edge_attr)
        return h, hs, edge_attrs
