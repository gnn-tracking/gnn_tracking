from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from gnn_tracking.models.dynamic_edge_conv import DynamicEdgeConv
from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP
from gnn_tracking.models.resin import ResIN


class INConvBlock(nn.Module):
    def __init__(
        self,
        indim,
        h_dim,
        e_dim,
        L,
        k,
        hidden_dim=100,
    ):
        super().__init__()
        self.indim = indim
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.L = L
        self.k = k
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_dim

        self.node_encoder = MLP(2 * indim, self.h_dim, hidden_dim=hidden_dim, L=1)
        self.edge_conv = DynamicEdgeConv(self.node_encoder, aggr="add", k=k)
        self.edge_encoder = MLP(2 * self.h_dim, self.e_dim, hidden_dim=hidden_dim, L=1)
        layers = []
        for _ in range(L):
            layers.append(
                IN(
                    self.h_dim,
                    self.e_dim,
                    node_outdim=self.h_dim,
                    edge_outdim=self.e_dim,
                    node_hidden_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:

        h, edge_index = self.edge_conv(x)
        h = self.relu(h)
        edge_attr = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)
        edge_attr = self.relu(self.edge_encoder(edge_attr))

        # apply the track condenser
        for layer in self.layers:
            delta_h, edge_attr = layer(h, edge_index, edge_attr)
            h = alpha * h + (1 - alpha) * delta_h
        return h


class PointCloudTCN(nn.Module):
    def __init__(
        self,
        node_indim: int,
        h_dim=10,
        e_dim=10,
        h_outdim=5,
        hidden_dim=100,
        N_blocks=3,
        L=3,
    ):
        """

        Args:
            node_indim:
            h_dim:   node dimension in latent space
            e_dim: edge dimension in latent space
            h_outdim:  output dimension in clustering space
            hidden_dim:  hidden with of all nn.Linear layers
            N_blocks:  number of edge_conv + IN blocks
            L: message passing depth in each block
        """
        super().__init__()
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.relu = nn.ReLU()

        layers = [INConvBlock(node_indim, h_dim, e_dim, L=L, k=N_blocks)]
        for i in range(N_blocks):
            layers.append(INConvBlock(h_dim, h_dim, e_dim, L=L, k=N_blocks - i))
        self.layers = nn.ModuleList(layers)

        # modules to predict outputs
        self.B = MLP(self.h_dim, 1, hidden_dim, L=3)
        self.X = MLP(self.h_dim, h_outdim, hidden_dim, L=3)

    def forward(
        self,
        data: Data,
        alpha: float = 0.5,
    ) -> dict[str, Tensor]:

        # apply the edge classifier to generate edge weights
        h = data.x
        for layer in self.layers:
            h = layer(h)

        beta = torch.sigmoid(self.B(h))
        # protect against nans
        beta = beta + torch.ones_like(beta) * 10e-12

        h_out = self.X(h)
        return {"W": None, "H": h_out, "B": beta, "P": None}


class GraphTCN(nn.Module):
    def __init__(
        self,
        node_indim: int,
        edge_indim: int,
        h_dim=5,
        e_dim=4,
        h_outdim=2,
        hidden_dim=40,
        L_ec=3,
        L_hc=3,
        alpha_ec: float = 0.5,
        alpha_hc: float = 0.5,
    ):
        """

        Args:
            node_indim:
            edge_indim:
            h_dim: node dimension in latent space
            e_dim: edge dimension in latent space
            h_outdim: output dimension in clustering space
            hidden_dim: width of hidden layers in all perceptrons
            L_ec: message passing depth for edge classifier
            L_hc: message passing depth for track condenser
            alpha_ec: strength of residual connection for EC
            alpha_hc: strength of residual connection for HC
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

        # specify the track condenser
        self.hc_node_encoder = MLP(
            node_indim, h_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.hc_edge_encoder = MLP(
            edge_indim, e_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.hc_resin = ResIN.identical_in_layers(
            node_indim=h_dim,
            edge_indim=e_dim,
            node_outdim=h_dim,
            edge_outdim=e_dim,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
            alpha=alpha_hc,
            n_layers=L_hc,
        )

        # modules to predict outputs
        self.W = MLP(e_dim * (L_ec + 1), 1, hidden_dim, L=3)
        self.B = MLP(h_dim, 1, hidden_dim, L=3)
        self.X = MLP(h_dim, h_outdim, hidden_dim, L=3)
        self.P = IN(
            h_dim,
            e_dim * (L_hc + 1),
            node_outdim=1,
            edge_outdim=1,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
        )

    def forward(
        self,
        data: Data,
    ) -> dict[str, Tensor]:
        # apply the edge classifier to generate edge weights
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h_ec = self.relu(self.ec_node_encoder(x))
        edge_attr_ec = self.relu(self.ec_edge_encoder(edge_attr))
        h_ec, _, edge_attrs_ec = self.ec_resin(h_ec, edge_index, edge_attr_ec)

        # append edge weights as new edge features
        edge_attrs_ec = torch.cat(edge_attrs_ec, dim=1)
        edge_weights = torch.sigmoid(self.W(edge_attrs_ec))
        # edge_attr = torch.cat((edge_weights, edge_attr), dim=1)

        # apply edge weight threshold
        row, col = edge_index
        mask = (edge_weights > 0.5).squeeze()
        row, col = row[mask], col[mask]
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = edge_attr[mask]

        # apply the track condenser
        h_hc = self.relu(self.hc_node_encoder(x))
        edge_attr_hc = self.relu(self.hc_edge_encoder(edge_attr))
        h_hc, _, edge_attrs_hc = self.hc_resin(h_hc, edge_index, edge_attr_hc)
        beta = torch.sigmoid(self.B(h_hc))
        # protect against nans
        beta = beta + torch.ones_like(beta) * 10e-9

        h = self.X(h_hc)
        track_params, _ = self.P(h_hc, edge_index, torch.cat(edge_attrs_hc, dim=1))
        return {"W": edge_weights, "H": h, "B": beta, "P": track_params}
