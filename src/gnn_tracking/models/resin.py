from __future__ import annotations

from torch import Tensor, nn
from torch.nn.functional import relu

from gnn_tracking.models.interaction_network import InteractionNetwork as IN


class ResIN(nn.Module):
    def __init__(self, layers: list[nn.Module], alpha: float = 0.5):
        """Apply a list of layers in sequence. Built for interaction networks, but any
        network that returns a node feature tensor and an edge feature tensor should
        work.

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
        alpha: float = 0.5,
        n_layers=1,
        **kwargs,
    ) -> ResIN:
        """Create a ResIN with identical layers interaction network layers

        Args:
            alpha: Strength of the residual connection
            n_layers: Number of layers
            **kwargs: Keyword arguments to pass to the interaction network layers
        """
        layers = [IN(**kwargs) for _ in range(n_layers)]
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
