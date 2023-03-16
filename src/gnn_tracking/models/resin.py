from __future__ import annotations

import math

import torch
from torch import Tensor
from torch import Tensor as T
from torch import nn
from torch.nn.functional import relu

from gnn_tracking.models.interaction_network import InteractionNetwork
from gnn_tracking.models.mlp import MLP


@torch.jit.script
def _convex_combination(*, delta: T, residue: T, alpha_residue: float) -> T:
    return alpha_residue * residue + (1 - alpha_residue) * relu(delta)


_IDENTITY = nn.Identity()


def convex_combination(
    *,
    delta: T,
    residue: T,
    alpha_residue: float,
    residue_encoder: nn.Module = _IDENTITY,
) -> T:
    """Convex combination of ``relu(delta)`` and the residue."""
    if math.isclose(alpha_residue, 0.0):
        return relu(delta)
    assert 0 <= alpha_residue <= 1
    return _convex_combination(
        delta=delta, residue=residue_encoder(residue), alpha_residue=alpha_residue
    )


class ResIN(nn.Module):
    def __init__(
        self,
        layers: list[nn.Module],
        length_concatenated_edge_attrs: int,
        *,
        alpha: float = 0.5,
    ):
        """Apply a list of layers in sequence with residual connections for the nodes.
        Built for interaction networks, but any network that returns a node feature
        tensor and an edge feature tensor should
        work.

        Note that a ReLu activation function is applied to the node result of the
        layer.

        Args:
            layers: List of layers
            length_concatenated_edge_attrs: Length of the concatenated edge attributes
                (from all the different layers)
            alpha: Strength of the node embedding residual connection
        """
        super().__init__()
        self._layers = nn.ModuleList(layers)
        self._alpha = alpha
        #: Because of the residual connections, we need map the output of the previous
        #: layer to the dimension of the next layer (if they are different). This
        #: can be done with these encoders.
        self._residue_node_encoders = nn.ModuleList([nn.Identity() for _ in layers])
        self.length_concatenated_edge_attrs = length_concatenated_edge_attrs

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

        If the input/hidden/output dimensions for the nodes are not the same, MLPs are
        used to map the previous output for the residual connection.

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
        if n_layers == 1:
            first_layer = InteractionNetwork(
                node_indim=node_indim,
                edge_indim=edge_indim,
                node_outdim=node_outdim,
                edge_outdim=edge_outdim,
                node_hidden_dim=object_hidden_dim,
                edge_hidden_dim=relational_hidden_dim,
            )
            mod = cls([first_layer], edge_outdim, alpha=alpha)
            if node_indim != node_outdim:
                first_encoder = MLP(
                    input_size=node_indim,
                    output_size=node_outdim,
                    hidden_dim=node_outdim,
                    L=2,
                    include_last_activation=True,
                )
            else:
                first_encoder = nn.Identity()
            mod._residue_node_encoders = nn.ModuleList([first_encoder])
            return mod

        first_layer = InteractionNetwork(
            node_indim=node_indim,
            edge_indim=edge_indim,
            node_outdim=node_hidden_dim,
            edge_outdim=edge_hidden_dim,
            node_hidden_dim=object_hidden_dim,
            edge_hidden_dim=relational_hidden_dim,
        )
        if node_indim != node_hidden_dim:
            first_encoder = MLP(
                input_size=node_indim,
                output_size=node_hidden_dim,
                hidden_dim=node_hidden_dim,
                L=2,
                include_last_activation=True,
            )
        else:
            first_encoder = nn.Identity()
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
        hidden_encoders = [nn.Identity() for _ in hidden_layers]
        last_layer = InteractionNetwork(
            node_indim=node_hidden_dim,
            edge_indim=edge_hidden_dim,
            node_outdim=node_outdim,
            edge_outdim=edge_outdim,
            node_hidden_dim=object_hidden_dim,
            edge_hidden_dim=relational_hidden_dim,
        )
        if node_hidden_dim != node_outdim:
            last_encoder = MLP(
                input_size=node_hidden_dim,
                output_size=node_outdim,
                hidden_dim=node_outdim,
                L=2,
                include_last_activation=True,
            )
        else:
            last_encoder = nn.Identity()
        layers = [first_layer, *hidden_layers, last_layer]
        encoders = [first_encoder, *hidden_encoders, last_encoder]
        assert len(layers) == n_layers == len(encoders)
        length_concatenated_edge_attrs = edge_hidden_dim * (n_layers - 1) + edge_outdim
        mod = cls(layers, length_concatenated_edge_attrs, alpha=alpha)
        mod._residue_node_encoders = nn.ModuleList(encoders)
        return mod

    def forward(
        self, x, edge_index, edge_attr
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Forward pass

        Args:
            x: Node features
            edge_index:
            edge_attr: Edge features

        Returns:
            node embedding, node embedding at each layer (including the input and
            final node embedding), edge embedding at each layer (including the input)
        """
        edge_attrs = [edge_attr]
        xs = [x]
        for layer, re in zip(self._layers, self._residue_node_encoders):
            delta_x, edge_attr = layer(x, edge_index, edge_attr)
            x = convex_combination(
                delta=delta_x, residue=x, alpha_residue=self._alpha, residue_encoder=re
            )
            xs.append(x)
            edge_attrs.append(edge_attr)
        return x, xs, edge_attrs
