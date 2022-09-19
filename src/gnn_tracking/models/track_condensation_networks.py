from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from gnn_tracking.models.dynamic_edge_conv import DynamicEdgeConv
from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP


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
        super(INConvBlock, self).__init__()
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
        super(PointCloudTCN, self).__init__()
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
            e_dim:  edge dimension in latent space
            h_outdim: output dimension in clustering space
            hidden_dim: hidden with of all nn.Linear layers
            L_ec: message passing depth for edge classifier
            L_hc: message passing depth for track condenser
            alpha_ec: strength of residual connection for EC
            alpha_hc: strength of residual connection for HC
        """
        super(GraphTCN, self).__init__()
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.relu = nn.ReLU()
        self.alpha_ec = alpha_ec
        self.alpha_hc = alpha_hc

        # specify the edge classifier
        self.ec_node_encoder = MLP(node_indim, self.h_dim, hidden_dim=hidden_dim, L=1)
        self.ec_edge_encoder = MLP(edge_indim, self.e_dim, hidden_dim=hidden_dim, L=1)
        ec_layers = []
        for _ in range(L_ec):
            ec_layers.append(
                IN(
                    self.h_dim,
                    self.e_dim,
                    node_outdim=self.h_dim,
                    edge_outdim=self.e_dim,
                    node_hidden_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                )
            )
        self.ec_layers = nn.ModuleList(ec_layers)

        # specify the track condenser
        self.hc_node_encoder = MLP(node_indim, self.h_dim, hidden_dim=hidden_dim, L=1)
        self.hc_edge_encoder = MLP(edge_indim, self.e_dim, hidden_dim=hidden_dim, L=1)
        hc_layers = []
        for _ in range(L_hc):
            hc_layers.append(
                IN(
                    self.h_dim,
                    self.e_dim,
                    node_outdim=self.h_dim,
                    edge_outdim=self.e_dim,
                    node_hidden_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                )
            )
        self.hc_layers = nn.ModuleList(hc_layers)

        # modules to predict outputs
        self.W = MLP(self.e_dim * (L_ec + 1), 1, hidden_dim, L=1)
        self.B = MLP(self.h_dim, 1, hidden_dim, L=1)
        self.X = MLP(self.h_dim, h_outdim, hidden_dim, L=1)
        self.P = IN(
            self.h_dim,
            self.e_dim * (L_hc + 1),
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
        edge_attrs_ec = [edge_attr_ec]
        for layer in self.ec_layers:
            delta_h_ec, new_edge_attr_ec = layer(h_ec, edge_index, edge_attr_ec)
            h_ec = self.alpha_ec * h_ec + (1 - self.alpha_ec) * self.relu(delta_h_ec)
            edge_attrs_ec.append(new_edge_attr_ec)
            edge_attr_ec = new_edge_attr_ec

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
        edge_attrs_hc = [edge_attr_hc]
        for layer in self.hc_layers:
            delta_h_hc, new_edge_attr_hc = layer(h_hc, edge_index, edge_attr_hc)
            h_hc = self.alpha_hc * h_hc + (1 - self.alpha_hc) * self.relu(delta_h_hc)
            edge_attrs_hc.append(new_edge_attr_hc)
            edge_attr_hc = new_edge_attr_hc

        beta = torch.sigmoid(self.B(h_hc))
        # protect against nans
        beta = beta + torch.ones_like(beta) * 10e-6

        h = self.X(h_hc)
        track_params, _ = self.P(h_hc, edge_index, torch.cat(edge_attrs_hc, dim=1))
        return {"W": edge_weights, "H": h, "B": beta, "P": track_params}
