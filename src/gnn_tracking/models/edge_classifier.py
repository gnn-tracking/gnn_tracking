import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from models.interaction_network import InteractionNetwork

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

class EdgeClassifier(nn.Module):
    def __init__(self, node_indim, edge_indim):
        super(EdgeClassifier, self).__init__()
        self.IN = InteractionNetwork(node_indim, edge_indim, 
                                     node_outdim=3, edge_outdim=4, 
                                     hidden_size=120)
        self.W = MLP(4, 1, 40) 
        
    def forward(self, x: Tensor, edge_index: Tensor, 
                edge_attr: Tensor) -> Tensor:

        x1, edge_attr_1 = self.IN(x, edge_index, edge_attr)
        return torch.sigmoid(self.W(edge_attr))
        
