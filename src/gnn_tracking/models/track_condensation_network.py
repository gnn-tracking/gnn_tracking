import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from models.interaction_network import InteractionNetwork as IN

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

class TCN(nn.Module):
    def __init__(self, node_indim, edge_indim, hc_outdim,
                 predict_track_params=False):
        super(TCN, self).__init__()
        self.h_dim = 7
        self.encoder = nn.Linear(node_indim, self.h_dim)
        self.in_w1 = IN(self.h_dim, edge_indim, 
                        node_outdim=self.h_dim, edge_outdim=4, 
                        hidden_size=40)
        self.in_w2 = IN(self.h_dim, 4, 
                        node_outdim=self.h_dim, edge_outdim=4, 
                        hidden_size=40)
        self.in_w3 = IN(self.h_dim, 4, 
                        node_outdim=self.h_dim, edge_outdim=4,
                        hidden_size=40)
        self.in_c1 = IN(self.h_dim, 17, 
                        node_outdim=self.h_dim, edge_outdim=8, 
                        hidden_size=40)
        self.in_c2 = IN(self.h_dim, 8, 
                        node_outdim=self.h_dim, edge_outdim=8, 
                        hidden_size=40)
        self.in_c3 = IN(self.h_dim, 8, 
                        node_outdim=self.h_dim, edge_outdim=8, 
                        hidden_size=40)
        
        self.W = MLP(16, 1, 40)
        self.B = MLP(self.h_dim, 1, 60)
        self.X = MLP(self.h_dim, hc_outdim, 80)
        
        if predict_track_params:
            self.p1 = IN(self.h_dim, 8, node_outdim=3, edge_outdim=3,
                         hidden_size=40)
            self.p2 = IN(3, 3, 3, 3, hidden_size=40)
            self.p3 = IN(3, 3, 3, 3,
                         hidden_size=40)
            #self.P = MLP(self.h_dim, 2, 80)
            #self.Q = MLP(self.h_dim, 1, 20)
        self.predict_track_params = predict_track_params
        
    def forward(self, x: Tensor, edge_index: Tensor, 
                edge_attr: Tensor) -> Tensor:
        
        # re-embed the graph twice with add aggregation
        h = self.encoder(x)
        h1, edge_attr_1 = self.in_w1(h, edge_index, edge_attr)
        h2, edge_attr_2 = self.in_w2(h+h1, edge_index, edge_attr_1)
        h3, edge_attr_3 = self.in_w3(h+h2, edge_index, edge_attr_2)
        
        # combine all edge features, use to predict edge weights
        initial_edge_attr = torch.cat([edge_attr, edge_attr_1, 
                                       edge_attr_2, edge_attr_3], dim=1)
        edge_weights = torch.sigmoid(self.W(initial_edge_attr))

        # combine edge weights with original edge features
        edge_attr_w = torch.cat([edge_weights, 
                                 initial_edge_attr], dim=1)

        hc1, edge_attr_c1 = self.in_c1(h+h3, edge_index, 
                                       edge_attr_w)
        hc2, edge_attr_c2 = self.in_c2(h+hc1, edge_index, 
                                       edge_attr_c1)
        hc3, edge_attr_c3 = self.in_c3(h+hc2, edge_index, 
                                       edge_attr_c2)
        hc3 = hc3+h
        beta = torch.sigmoid(self.B(hc3))
        hc = self.X(hc3)
        if self.predict_track_params:
            p1, edge_attr_p1 = self.p1(hc3, edge_index, edge_attr_c3)
            p2, edge_attr_p2 = self.p2(p1, edge_index, edge_attr_p1)
            p3, edge_attr_p3 = self.p3(p2, edge_index, edge_attr_p2)
            #q = 2 * torch.sigmoid(self.Q(hc3)) - 1
            #p = torch.cat((p, q), dim=1)
            return edge_weights, hc, beta, p3

        return edge_weights, hc, beta
        
