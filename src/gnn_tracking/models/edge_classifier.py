from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP


class EdgeClassifier(nn.Module):
    def __init__(
        self,
        node_indim,
        edge_indim,
        L=4,
        node_latentdim=8,
        edge_latentdim=12,
        r_hidden_size=32,
        o_hidden_size=32,
    ):
        super(EdgeClassifier, self).__init__()
        self.node_encoder = MLP(node_indim, node_latentdim, 64, L=1)
        self.edge_encoder = MLP(edge_indim, edge_latentdim, 64, L=1)
        gnn_layers = []
        for _l in range(L):
            # fixme: Wrong parameters?
            gnn_layers.append(
                IN(
                    node_latentdim,
                    edge_latentdim,
                    node_outdim=node_latentdim,
                    edge_outdim=edge_latentdim,
                    edge_hidden_dim=r_hidden_size,
                    node_hidden_dim=o_hidden_size,
                )
            )
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.W = MLP(edge_latentdim, 1, 32, L=2)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            node_latent, edge_latent = layer(node_latent, edge_index, edge_latent)
        edge_weights = torch.sigmoid(self.W(edge_latent))
        return edge_weights
