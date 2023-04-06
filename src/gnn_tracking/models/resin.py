from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor as T
from torch import nn
from torch.nn.functional import relu

from gnn_tracking.models.interaction_network import InteractionNetwork


@torch.jit.script
def _convex_combination(
    *,
    delta: T,
    residue: T,
    alpha_residue: float,
) -> T:
    """Helper function for JIT compilation"""
    assert 0 <= alpha_residue <= 1
    return alpha_residue * residue + (1 - alpha_residue) * delta


def convex_combination(
    *,
    delta: T,
    residue: T | None,
    alpha_residue: float,
) -> T:
    """Convex combination of delta and residue"""
    if residue is None:
        return delta
    if math.isclose(alpha_residue, 0.0):
        return delta
    return _convex_combination(
        delta=delta, residue=residue, alpha_residue=alpha_residue
    )


class ResidualNetwork(ABC, nn.Module):
    def __init__(
        self,
        layers: list[nn.Module],
        *,
        alpha_node: float = 0.5,
        alpha_edge: float = 0.5,
    ):
        """Apply a list of layers in sequence with residual connections for the nodes.
        This is an abstract base class that does not contain code for the type of
        residual connections.

        Use one of the subclasses below.

        Args:
            layers: List of layers
            alpha_node: Strength of the node embedding residual connection
            alpha_edge: Strength of the edge embedding residual connection
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self._alpha_node = alpha_node
        self._alpha_edge = alpha_edge

    def forward(self, x, edge_index, edge_attr) -> tuple[T, T, list[T], list[T]]:
        """Forward pass

        Args:
            x: Node features
            edge_index:
            edge_attr: Edge features

        Returns:
            node embedding, edge_embedding, concatenated node embeddings from all levels
            (including x), concatenated edge embeddings from all levels (include
            ``edge_attr``)
        """
        return self._forward(x, edge_index, edge_attr)

    @abstractmethod
    def _forward(self, x, edge_index, edge_attr) -> tuple[T, T, list[T], list[T]]:
        pass


class Skip1ResidualNetwork(ResidualNetwork):
    def _forward(self, x, edge_index, edge_attr) -> tuple[T, T, list[T], list[T]]:
        xs = [x]
        edge_attrs = [edge_attr]
        for layer in self.layers:
            delta_x, delta_edge_attr = layer(x, edge_index, edge_attr)
            x = convex_combination(
                delta=relu(delta_x),
                residue=x,
                alpha_residue=self._alpha_node,
            )
            edge_attr = convex_combination(
                delta=relu(delta_edge_attr),
                residue=edge_attr,
                alpha_residue=self._alpha_edge,
            )
            xs.append(x)
            edge_attrs.append(edge_attr)
        return x, edge_attr, xs, edge_attrs


class Skip2ResidualNetwork(ResidualNetwork):
    def __init__(
        self,
        layers: list[nn.Module],
        *,
        node_dim: int,
        edge_dim: int,
        add_bn: bool = True,
        **kwargs,
    ):
        """Residual network with skip connections every two layers.

        Args:
            layers: List of layers
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            add_bn: Add batch norms
            **kwargs: Arguments to `ResidualNetwork`
        """
        if not len(layers) % 2 == 0:
            raise ValueError("Only even number of layers allowed at the moment")
        super().__init__(layers=layers, **kwargs)
        _node_batch_norms = []
        _edge_batch_norms = []
        for _ in range(len(layers)):
            if add_bn:
                _node_batch_norms.append(nn.BatchNorm1d(node_dim))
                _edge_batch_norms.append(nn.BatchNorm1d(edge_dim))
            else:
                _node_batch_norms.append(nn.Identity())
                _edge_batch_norms.append(nn.Identity())
        self._node_batch_norms = nn.ModuleList(_node_batch_norms)
        self._edge_batch_norms = nn.ModuleList(_edge_batch_norms)

    def _forward(self, x, edge_index, edge_attr) -> tuple[T, T, list[T], list[T]]:
        xs = [x]
        edge_attrs = [edge_attr]
        for i_layer_pair in range(len(self.layers) // 2):
            i0 = 2 * i_layer_pair
            hidden_x, hidden_edge_attr = self.layers[i0](
                relu(self._node_batch_norms[i0](x)),
                edge_index,
                relu(self._edge_batch_norms[i0](edge_attr)),
            )
            i1 = 2 * i_layer_pair + 1
            delta_x, delta_edge_attr = self.layers[i1](
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
            xs.append(x)
            edge_attrs.append(edge_attr)
        return x, edge_attr, xs, edge_attrs


class SkipTopResidualNetwork(ResidualNetwork):
    def __init__(
        self,
        layers: list[nn.Module],
        connect_to=1,
        **kwargs,
    ):
        """Residual network with skip connections to the top layer.

        Args:
            layers: List of layers
            connect_to: Layer to which to add the skip connection. 0 means to the
                input, 1 means to the output of the first layer, etc.
            **kwargs: Arguments to `ResidualNetwork`
        """
        assert connect_to <= len(layers)
        super().__init__(layers=layers, **kwargs)
        self._residual_layer = connect_to

    def _forward(self, x, edge_index, edge_attr) -> tuple[T, T, list[T], list[T]]:
        xs = [x]
        edge_attrs = [edge_attr]
        x_residue = None
        edge_attr_residue = None
        for i_layer in range(len(self.layers)):
            if i_layer == self._residual_layer:
                x_residue = x
                edge_attr_residue = edge_attr
            delta_x, delta_edge_attr = self.layers[i_layer](x, edge_index, edge_attr)
            x = convex_combination(
                delta=relu(delta_x), residue=x_residue, alpha_residue=self._alpha_node
            )
            edge_attr = convex_combination(
                delta=relu(delta_edge_attr),
                residue=edge_attr_residue,
                alpha_residue=self._alpha_edge,
            )
            xs.append(x)
            edge_attrs.append(edge_attr)
        return x, edge_attr, xs, edge_attrs


RESIDUALNETWORKS_BY_NAME: dict[str, Any] = {
    "skip1": Skip1ResidualNetwork,
    "skip2": Skip2ResidualNetwork,
    "skip_top": SkipTopResidualNetwork,
}


class ResIN(nn.Module):
    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        object_hidden_dim=40,
        relational_hidden_dim=40,
        alpha_node: float = 0.5,
        alpha_edge: float = 0.5,
        n_layers=1,
        residual_type: str = "skip1",
        residual_kwargs: dict | None = None,
    ):
        """Create a ResIN with identical layers of interaction networks

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            object_hidden_dim: Hidden dimension for the object model MLP
            relational_hidden_dim: Hidden dimension for the relational model MLP
            alpha_node: Strength of the node residual connection
            alpha_edge: Strength of the edge residual connection
            n_layers: Total number of layers
            residual_type: Type of residual network. Options are 'skip1', 'skip2'.
            residual_kwargs: Additional arguments to the residual network (can depend on
                the residual type)
        """
        super().__init__()
        if residual_kwargs is None:
            residual_kwargs = {}
        layers = [
            InteractionNetwork(
                node_indim=node_dim,
                edge_indim=edge_dim,
                node_outdim=node_dim,
                edge_outdim=edge_dim,
                node_hidden_dim=object_hidden_dim,
                edge_hidden_dim=relational_hidden_dim,
            )
            for _ in range(n_layers)
        ]

        if residual_type == "skip2":
            residual_kwargs["node_dim"] = node_dim
            residual_kwargs["edge_dim"] = edge_dim

        network = RESIDUALNETWORKS_BY_NAME[residual_type](
            layers,
            alpha_node=alpha_node,
            alpha_edge=alpha_edge,
            **residual_kwargs,
        )
        self.network = network
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self._residual_type = residual_type

    @property
    def concat_node_embeddings_length(self) -> int:
        """Length of the concatenated node embeddings from all intermediate layers.
        Or in other words: `self.forward()[2].shape[1]`
        """
        if self._residual_type == "skip2":
            return self.node_dim * (len(self.network.layers) // 2 + 1)
        return self.node_dim * (len(self.network.layers) + 1)

    @property
    def concat_edge_embeddings_length(self) -> int:
        """Length of the concatenated edge embeddings from all intermediate layers.
        Or in other words: `self.forward()[3].shape[1]`
        """
        if self._residual_type == "skip2":
            return self.edge_dim * (len(self.network.layers) // 2 + 1)
        return self.edge_dim * (len(self.network.layers) + 1)

    def forward(self, x, edge_index, edge_attr) -> tuple[T, T, list[T], list[T]]:
        return self.network.forward(x, edge_index, edge_attr)
