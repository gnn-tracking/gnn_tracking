import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU
from torch.nn import Sequential as Seq
from torch.nn import Sigmoid
import torch_geometric
from torch_geometric.nn import MessagePassing
from models.mlp import MLP
from models.interaction_network import InteractionNetwork as IN

class EdgeClassifier(nn.Module):
    def __init__(self, node_indim, edge_indim, L=4,
                 node_latentdim=8, edge_latentdim=12,
                 r_hidden_size=32, o_hidden_size=32):
        super(EdgeClassifier, self).__init__()
        self.node_encoder = MLP(node_indim, node_latentdim, 64, L=1)
        self.edge_encoder = MLP(edge_indim, edge_latentdim, 64, L=1)
        gnn_layers = []
        for l in range(L):
            gnn_layers.append(IN(node_latentdim, edge_latentdim,
                                 node_outdim=node_latentdim, edge_outdim=edge_latentdim,
                                 relational_hidden_size=r_hidden_size, 
                                 object_hidden_size=o_hidden_size))
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.W = MLP(edge_latentdim, 1, 32, L=2)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        for l in self.gnn_layers:
            node_latent, edge_latent = l(node_latent, edge_index, edge_latent)
        edge_weights = torch.sigmoid(self.W(edge_latent))
        return edge_weights
