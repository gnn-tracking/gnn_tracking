from __future__ import annotations

import torch
import torch.nn as nn
from models.interaction_network import InteractionNetwork as IN
from models.mlp import MLP
from torch import Tensor


class PointCloudTCN(nn.Module):
    def __init__(
        self, node_indim, edge_indim, hc_outdim, hidden_dim, predict_track_params=False
    ):
        super(PointCloudTCN, self).__init__()
        self.h_dim = 7
        self.encoder = nn.Linear(node_indim, self.h_dim)
        self.in_w1 = IN(
            self.h_dim,
            edge_indim,
            node_outdim=self.h_dim,
            edge_outdim=4,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
        )
        self.in_w2 = IN(
            self.h_dim,
            4,
            node_outdim=self.h_dim,
            edge_outdim=4,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
        )
        self.in_w3 = IN(
            self.h_dim,
            4,
            node_outdim=self.h_dim,
            edge_outdim=4,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
        )
        self.in_c1 = IN(
            self.h_dim,
            17,
            node_outdim=self.h_dim,
            edge_outdim=8,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
        )
        self.in_c2 = IN(
            self.h_dim,
            8,
            node_outdim=self.h_dim,
            edge_outdim=8,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
        )
        self.in_c3 = IN(
            self.h_dim,
            8,
            node_outdim=self.h_dim,
            edge_outdim=8,
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
        )

        self.W = MLP(16, 1, 40)
        self.B = MLP(self.h_dim, 1, 60)
        self.X = MLP(self.h_dim, hc_outdim, 80)

        if predict_track_params:
            self.p1 = IN(self.h_dim, 8, node_outdim=3, edge_outdim=3, hidden_size=40)
            self.p2 = IN(3, 3, 3, 3, hidden_size=40)
            self.p3 = IN(3, 3, 3, 3, hidden_size=40)
            # self.P = MLP(self.h_dim, 2, 80)
            # self.Q = MLP(self.h_dim, 1, 20)
        self.predict_track_params = predict_track_params

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:

        # re-embed the graph twice with add aggregation
        h = self.encoder(x)
        h1, edge_attr_1 = self.in_w1(h, edge_index, edge_attr)
        h2, edge_attr_2 = self.in_w2(h + h1, edge_index, edge_attr_1)
        h3, edge_attr_3 = self.in_w3(h + h2, edge_index, edge_attr_2)

        # combine all edge features, use to predict edge weights
        initial_edge_attr = torch.cat(
            [edge_attr, edge_attr_1, edge_attr_2, edge_attr_3], dim=1
        )
        edge_weights = torch.sigmoid(self.W(initial_edge_attr))

        # combine edge weights with original edge features
        edge_attr_w = torch.cat([edge_weights, initial_edge_attr], dim=1)

        hc1, edge_attr_c1 = self.in_c1(h + h3, edge_index, edge_attr_w)
        hc2, edge_attr_c2 = self.in_c2(h + hc1, edge_index, edge_attr_c1)
        hc3, edge_attr_c3 = self.in_c3(h + hc2, edge_index, edge_attr_c2)
        hc3 = hc3 + h
        beta = torch.sigmoid(self.B(hc3))
        hc = self.X(hc3)
        if self.predict_track_params:
            p1, edge_attr_p1 = self.p1(hc3, edge_index, edge_attr_c3)
            p2, edge_attr_p2 = self.p2(p1, edge_index, edge_attr_p1)
            p3, edge_attr_p3 = self.p3(p2, edge_index, edge_attr_p2)
            # q = 2 * torch.sigmoid(self.Q(hc3)) - 1
            # p = torch.cat((p, q), dim=1)
            return edge_weights, hc, beta, p3

        return edge_weights, hc, beta


import torch
import torch.nn as nn
from models.interaction_network import InteractionNetwork as IN
from models.mlp import MLP
from torch import Tensor


class GraphTCN(nn.Module):
    def __init__(
        self,
        node_indim,
        edge_indim,
        e_dim=4,
        h_dim=5,
        h_outdim=2,
        hidden_dim=40,
        predict_track_params=False,
        L=3,
        C=3,
    ):
        super(GraphTCN, self).__init__()
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.node_encoder = MLP(node_indim, self.h_dim, hidden_dim=hidden_dim, L=1)
        self.edge_encoder = MLP(edge_indim, self.e_dim, hidden_dim=hidden_dim, L=1)

        # define edge classifier layers
        ec_layers = []
        for l in range(1, L - 1):
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

        # define the condensation layers
        hc_layers = []
        for l in range(1, C - 1):
            hc_layers.append(
                IN(
                    self.h_dim,
                    self.e_dim + 1,
                    node_outdim=self.h_dim,
                    edge_outdim=self.e_dim + 1,
                    node_hidden_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                )
            )
        self.hc_layers = nn.ModuleList(hc_layers)

        self.relu = nn.ReLU()
        self.W = MLP(self.e_dim, 1, 40, L=3)
        self.B = MLP(self.h_dim, 1, 40, L=3)
        self.X = MLP(self.h_dim, h_outdim, 40)

        if predict_track_params:
            self.p1 = IN(
                self.h_dim,
                8,
                node_outdim=3,
                edge_outdim=3,
                node_hidden_dim=hidden_dim,
                edge_hidden_dim=hidden_dim,
            )
            self.p2 = IN(3, 3, 3, 3, hidden_size=40)
            self.p3 = IN(3, 3, 3, 3, hidden_size=40)
        self.predict_track_params = predict_track_params

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:

        # re-embed the graph L times with add aggregation
        h = self.relu(self.node_encoder(x))
        edge_attr = self.relu(self.edge_encoder(edge_attr))
        for layer in self.ec_layers:
            delta_h, delta_edge_attr = layer(h, edge_index, edge_attr)
            h = h + delta_h
            edge_attr = edge_attr + delta_edge_attr

        # append edge weights as new edge features
        edge_weights = torch.sigmoid(self.W(edge_attr))
        edge_attr = torch.cat((edge_weights, edge_attr), dim=1)

        for layer in self.hc_layers:
            delta_h, edge_attr = layer(h, edge_index, edge_attr)
            h = h + delta_h

        beta = torch.sigmoid(self.B(h))
        h = self.X(h)
        return edge_weights, h, beta
