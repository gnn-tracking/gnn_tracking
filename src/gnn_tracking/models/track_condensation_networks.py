from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gnn_tracking.models.dynamic_edge_conv import DynamicEdgeConv
from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP


class PointCloudTCN(nn.Module):
    def __init__(
        self,
        node_indim,
        h_dim=5,  # node dimension in latent space
        e_dim=8,  # edge dimension in latent space
        h_outdim=2,  # output dimension in clustering space
        hidden_dim=40,  # hidden with of all nn.Linear layers
        L_hc=3,  # message passing depth for track condenser
        k=4,  # number of neighbors to connect in latent space
    ):
        super(PointCloudTCN, self).__init__()
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.relu = nn.ReLU()

        # specify the edge builder / classifier
        self.node_encoder = MLP(2 * node_indim, self.h_dim, hidden_dim=hidden_dim, L=1)
        self.edge_conv = DynamicEdgeConv(self.node_encoder, aggr="max", k=k)
        self.edge_encoder = MLP(2 * self.h_dim, self.e_dim, hidden_dim=hidden_dim, L=1)
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
        self.B = MLP(self.h_dim, 1, hidden_dim, L=3)
        self.X = MLP(self.h_dim, h_outdim, hidden_dim, L=3)
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
        x: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:

        # apply the edge classifier to generate edge weights
        h, edge_index = self.edge_conv(x)
        h = self.relu(h)
        edge_attr = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)
        edge_attr = self.relu(self.edge_encoder(edge_attr))

        # apply the track condenser
        edge_attrs = [edge_attr]
        for layer in self.hc_layers:
            delta_h, new_edge_attr = layer(h, edge_index, edge_attr)
            h = alpha * h + (1 - alpha) * delta_h
            edge_attrs.append(new_edge_attr)
            edge_attr = new_edge_attr

        beta = torch.sigmoid(self.B(h))
        # protect against nans
        beta = beta + torch.ones_like(beta) * 10e-6

        h_out = self.X(h)
        track_params, _ = self.P(h, edge_index, torch.cat(edge_attrs, dim=1))
        return h_out, beta, track_params


class GraphTCN(nn.Module):
    def __init__(
        self,
        node_indim,
        edge_indim,
        h_dim=5,  # node dimension in latent space
        e_dim=4,  # edge dimension in latent space
        h_outdim=2,  # output dimension in clustering space
        hidden_dim=40,  # hidden with of all nn.Linear layers
        L_ec=3,  # message passing depth for edge classifier
        L_hc=3,  # message passing depth for track condenser
    ):
        super(GraphTCN, self).__init__()
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.relu = nn.ReLU()

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
        self.hc_edge_encoder = MLP(
            edge_indim + 1, self.e_dim, hidden_dim=hidden_dim, L=1
        )
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
        self.W = MLP(self.e_dim * (L_ec + 1), 1, hidden_dim, L=3)
        self.B = MLP(self.h_dim, 1, hidden_dim, L=3)
        self.X = MLP(self.h_dim, h_outdim, hidden_dim, L=3)
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
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        alpha_ec: float = 0.5,
        alpha_hc: float = 0.5,
    ) -> Tensor:

        # apply the edge classifier to generate edge weights
        h_ec = self.relu(self.ec_node_encoder(x))
        edge_attr_ec = self.relu(self.ec_edge_encoder(edge_attr))
        edge_attrs_ec = [edge_attr_ec]
        for layer in self.ec_layers:
            delta_h_ec, new_edge_attr_ec = layer(h_ec, edge_index, edge_attr_ec)
            h_ec = alpha_ec * h_ec + (1 - alpha_ec) * delta_h_ec
            edge_attrs_ec.append(new_edge_attr_ec)
            edge_attr_ec = new_edge_attr_ec

        # append edge weights as new edge features
        edge_attrs_ec = torch.cat(edge_attrs_ec, dim=1)
        edge_weights = torch.sigmoid(self.W(edge_attrs_ec))
        edge_attr = torch.cat((edge_weights, edge_attr), dim=1)

        # apply the track condenser
        h_hc = self.relu(self.hc_node_encoder(x))
        edge_attr_hc = self.relu(self.hc_edge_encoder(edge_attr))
        edge_attrs_hc = [edge_attr_hc]
        for layer in self.hc_layers:
            delta_h_hc, new_edge_attr_hc = layer(h_hc, edge_index, edge_attr_hc)
            h_hc = alpha_hc * h_hc + (1 - alpha_hc) * delta_h_hc
            edge_attrs_hc.append(new_edge_attr_hc)
            edge_attr_hc = new_edge_attr_hc

        beta = torch.sigmoid(self.B(h_hc))
        # protect against nans
        beta = beta + torch.ones_like(beta) * 10e-6

        h = self.X(h_hc)
        track_params, _ = self.P(h_hc, edge_index, torch.cat(edge_attrs_hc, dim=1))
        return edge_weights, h, beta, track_params
