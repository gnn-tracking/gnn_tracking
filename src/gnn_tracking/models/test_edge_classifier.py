from __future__ import annotations

import torch

from gnn_tracking.models.edge_classifier import PerfectEdgeClassification
from gnn_tracking.utils.seeds import fix_seeds


class MockData:
    def __init__(self, y: list[bool] | torch.Tensor, pt: list[float] | None = None):
        self.y = torch.Tensor(y)
        if pt is not None:
            self.pt = torch.Tensor(pt)
        else:
            self.pt = torch.full_like(self.y, 0.5)


def test_perfect_edge_classification():
    pec = PerfectEdgeClassification()
    y = [True, False, True, False]
    d = MockData(y=y)
    assert (pec.forward(d) == torch.Tensor(y)).all()


def test_perfect_edge_classification_pt_thld():
    pec = PerfectEdgeClassification(false_below_pt=0.5)
    y = [True, False, True, False]
    d = MockData(y=y, pt=[0, 0, 1, 1])
    assert (pec.forward(d) == torch.Tensor([False, False, True, False])).all()


def test_perfect_edge_classification_tpr_tnr():
    fix_seeds()
    y = torch.full((100,), True)
    d = MockData(y=y)
    pec = PerfectEdgeClassification(tpr=0.5)
    assert 45 < pec.forward(d).sum() < 55
