from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP


class PointCloudTCN(nn.Module):
    def __init__(
        self,
        node_indim: int,
        edge_indim: int,
        hc_outdim: int,
        hidden_dim: int,
        predict_track_params=False,
    ):
        super().__init__()
        self.h_dim = 7
        self.encoder = nn.Linear(node_indim, self.h_dim)
        shared_kwargs = dict(
            node_hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
            node_indim=self.h_dim,
            node_outdim=self.h_dim,
        )
        self.in_w1 = IN(edge_indim=edge_indim, edge_outdim=4, **shared_kwargs)
        self.in_w2 = IN(edge_indim=4, edge_outdim=4, **shared_kwargs)
        self.in_w3 = IN(edge_indim=4, edge_outdim=4, **shared_kwargs)
        self.in_c1 = IN(edge_indim=17, edge_outdim=8, **shared_kwargs)
        self.in_c2 = IN(edge_indim=8, edge_outdim=8, **shared_kwargs)
        self.in_c3 = IN(edge_indim=8, edge_outdim=8, **shared_kwargs)

        self.W = MLP(16, 1, 40)
        self.B = MLP(self.h_dim, 1, 60)
        self.X = MLP(self.h_dim, hc_outdim, 80)

        if predict_track_params:
            hidden_dim_spec = dict(node_hidden_dim=40, edge_hidden_dim=40)
            self.p1 = IN(self.h_dim, 8, node_outdim=3, edge_outdim=3, **hidden_dim_spec)
            self.p2 = IN(3, 3, 3, 3, **hidden_dim_spec)
            self.p3 = IN(3, 3, 3, 3, **hidden_dim_spec)
            # self.P = MLP(self.h_dim, 2, 80)
            # self.Q = MLP(self.h_dim, 1, 20)
        self.predict_track_params = predict_track_params

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):

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
        h = self.X(h_hc)
        track_params, _ = self.P(h_hc, edge_index, torch.cat(edge_attrs_hc, dim=1))
        return edge_weights, h, beta, track_params
