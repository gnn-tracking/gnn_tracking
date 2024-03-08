import torch
from torch_geometric.data import Data

from gnn_tracking.models.graph_construction import (
    GraphConstructionFCNN,
    GraphConstructionHeteroEncResFCNN,
    GraphConstructionHeteroResFCNN,
)


def test_graphconstructionfcnn():
    model = GraphConstructionFCNN(
        in_dim=3,
        hidden_dim=4,
        out_dim=5,
        depth=2,
        alpha=0.7,
    )
    model(
        Data(x=torch.rand(7, 3)),
    )


def test_graphconstructionheteroresfcnn():
    model = GraphConstructionHeteroResFCNN(
        in_dim=3,
        hidden_dim=4,
        out_dim=5,
        depth=2,
        alpha=0.7,
    )
    model(
        Data(x=torch.rand(7, 3), layer=torch.tensor([0, 1, 2, 20, 21, 22, 23])),
    )


def test_graphconstructionheteroencresfcnn():
    model = GraphConstructionHeteroEncResFCNN(
        in_dim=3,
        hidden_dim=4,
        hidden_dim_enc=3,
        out_dim=5,
        depth=2,
        alpha=0.7,
        depth_enc=2,
    )
    model(
        Data(x=torch.rand(7, 3), layer=torch.tensor([0, 1, 2, 20, 21, 22, 23])),
    )
