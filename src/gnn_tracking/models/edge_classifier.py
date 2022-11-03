from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP
from gnn_tracking.models.resin import ResIN


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


class ECForGraphTCN(nn.Module):
    def __init__(
        self,
        *,
        node_indim: int,
        edge_indim: int,
        h_dim=5,
        e_dim=4,
        hidden_dim=40,
        L_ec=3,
        alpha_ec: float = 0.5,
    ):
        """Edge classification step to be used for Graph Track Condensor network
        (Graph TCN)

        Args:
            node_indim: Node feature dim
            edge_indim: Edge feature dim
            h_dim: node dimension in latent space
            e_dim: edge dimension in latent space
            hidden_dim: width of hidden layers in all perceptrons
            L_ec: message passing depth for edge classifier
            alpha_ec: strength of residual connection for EC
        """
        super().__init__()
        self.relu = nn.ReLU()

        # specify the edge classifier
        self.ec_node_encoder = MLP(
            node_indim, h_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.ec_edge_encoder = MLP(
            edge_indim, e_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.ec_resin = ResIN.identical_in_layers(
            node_indim=h_dim,
            edge_indim=e_dim,
            node_outdim=h_dim,
            edge_outdim=e_dim,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
            alpha=alpha_ec,
            n_layers=L_ec,
        )

        self.W = MLP(e_dim * (L_ec + 1), 1, hidden_dim, L=3)

    def forward(
        self,
        data: Data,
    ) -> Tensor:
        # apply the edge classifier to generate edge weights
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h_ec = self.relu(self.ec_node_encoder(x))
        edge_attr_ec = self.relu(self.ec_edge_encoder(edge_attr))
        h_ec, _, edge_attrs_ec = self.ec_resin(h_ec, edge_index, edge_attr_ec)

        # append edge weights as new edge features
        edge_attrs_ec = torch.cat(edge_attrs_ec, dim=1)
        edge_weights = torch.sigmoid(self.W(edge_attrs_ec))
        return edge_weights


class PerfectEdgeClassification(nn.Module):
    def __init__(self, tpr=1.0, tnr=1.0):
        """An edge classifier that is perfect because it uses the truth information.
        If TPR or TNR is not 1.0, noise is added to the truth information.

        Args:
            tpr: True positive rate
            tnr: False positive rate
        """
        super().__init__()
        if not np.isclose(tpr, 1.0):
            raise NotImplementedError("Handling of arbitrary TPR not yet implemented")
        if not np.isclose(tnr, 1.0):
            raise NotImplementedError("Handling of arbitrary TNR not yet implemented")

    def forward(self, data: Data) -> Tensor:
        return data.y
