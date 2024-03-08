import torch

from gnn_tracking.models.mlp import ResFCNN
from gnn_tracking.utils.asserts import assert_feat_dim


def test_resfcnn():
    model = ResFCNN(
        in_dim=3,
        hidden_dim=4,
        out_dim=5,
        depth=6,
        alpha=0.7,
        bias=True,
    )
    out = model(
        x=torch.rand(1, 3),
    )
    assert_feat_dim(out, 5)


def test_hetero_resfcnn():
    model = ResFCNN(
        in_dim=3,
        hidden_dim=4,
        out_dim=5,
        depth=6,
        alpha=0.7,
        bias=True,
    )
    out = model(
        x=torch.rand(1, 3),
        layer=torch.tensor([0, 1, 20]),
    )
    assert_feat_dim(out, 5)
