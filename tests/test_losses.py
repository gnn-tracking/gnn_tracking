from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from pytest import approx  # noqa: PT013
from torch.nn.functional import binary_cross_entropy
from typing_extensions import TypeAlias

from gnn_tracking.metrics.losses import (
    LossClones,
)
from gnn_tracking.metrics.losses.ec import EdgeWeightBCELoss, binary_focal_loss
from gnn_tracking.metrics.losses.metric_learning import (
    GraphConstructionHingeEmbeddingLoss,
)
from gnn_tracking.metrics.losses.oc import (
    CondensationLossRG,
    CondensationLossTiger,
    ObjectLoss,
)
from gnn_tracking.preprocessing.point_cloud_builder import get_truth_edge_index
from gnn_tracking.utils.dictionaries import to_floats

T: TypeAlias = torch.Tensor

# ruff: noqa: T201


@dataclass
class MockData:
    beta: T
    x: T
    particle_id: T
    pred: T
    truth: T
    pt: T
    eta: T
    reconstructable: T
    batch: T
    true_edge_index: T


def generate_test_data(
    n_nodes=1000, n_particles=250, n_x_features=3, rng=None
) -> MockData:
    if rng is None:
        rng = np.random.default_rng()

    pid = torch.from_numpy(rng.choice(np.arange(n_particles), size=n_nodes))
    pid_unique = torch.unique(pid)
    pt_pid = torch.from_numpy(2 * rng.random(len(pid_unique)))
    pt = pt_pid[pid]
    eta_pid = torch.from_numpy(8 * (rng.random(len(pid_unique)) - 0.5))
    eta = eta_pid[pid]
    reco_pid = torch.from_numpy(rng.choice([0.0, 1.0], size=len(pid_unique)))
    reco = reco_pid[pid]

    return MockData(
        beta=torch.from_numpy(rng.random(n_nodes)),
        x=torch.from_numpy(rng.random((n_nodes, n_x_features))),
        particle_id=pid,
        pred=torch.from_numpy(rng.choice([0.0, 1.0], size=(n_nodes, 1))),
        truth=torch.from_numpy(rng.choice([0.0, 1.0], size=(n_nodes, 1))),
        pt=pt,
        eta=eta,
        reconstructable=reco,
        batch=torch.zeros_like(reco),
        true_edge_index=torch.from_numpy(get_truth_edge_index(pid.numpy())),
    )


td1 = generate_test_data(50, n_particles=3, rng=np.random.default_rng(seed=0))
td2 = generate_test_data(100, n_particles=10, rng=np.random.default_rng(seed=0))


def get_condensation_loss(td: MockData, *, strategy="tiger", **kwargs) -> float:
    if strategy == "tiger":
        loss_fct = CondensationLossTiger(
            **kwargs,
        )
    elif strategy == "rg":
        loss_fct = CondensationLossRG(
            **kwargs,
        )
    else:
        raise ValueError
    return to_floats(
        loss_fct(
            beta=td.beta,
            x=td.x,
            particle_id=td.particle_id,
            reconstructable=td.reconstructable,
            pt=td.pt,
            eta=td.eta,
        ).loss_dct
    )


def get_object_loss(td: MockData, **kwargs) -> float:
    return (
        ObjectLoss(**kwargs)
        .object_loss(
            beta=td.beta, particle_id=td.particle_id, pred=td.pred, truth=td.truth
        )
        .item()
    )


_td1_c_losses = {
    "attractive": 0.48778231210119105,
    "repulsive": 35939197600.633316,
    "coward": 0.051056325062234675,
    "noise": 0.5346992111891886,
}
_td2_c_losses = {
    "attractive": 1.5953161268602611,
    "repulsive": 3.478838882898964,
    "coward": 0.03316374922649601,
    "noise": 0.564675177839844,
}


def test_pin_condensation_losses_tiger():
    assert get_condensation_loss(td1) == approx(_td1_c_losses)
    assert get_condensation_loss(td2) == approx(_td2_c_losses)


def test_pin_condensation_losses_rg():
    assert get_condensation_loss(td1, strategy="rg") == approx(_td1_c_losses)
    assert get_condensation_loss(td2, strategy="rg") == approx(_td2_c_losses)


def test_pin_object_loss_efficiency():
    assert get_object_loss(td1) == approx(0.4858411097284774)
    assert get_object_loss(td2) == approx(0.5769124284752167)


def test_pin_object_loss_purity():
    assert get_object_loss(td1, mode="purity") == approx(0.010453588032279765)
    assert get_object_loss(td2, mode="purity") == approx(0.00563383851854332)


def test_focal_loss_vs_bce():
    inpt = torch.rand(10)
    target = (torch.rand(10) > 0.5).float()
    assert binary_focal_loss(inpt=inpt, target=target, alpha=0.5, gamma=0.0) == approx(
        0.5
        * binary_cross_entropy(
            inpt,
            target,
        )
    )


def test_loss_clones():
    loss = EdgeWeightBCELoss()
    eclc = LossClones(
        loss,
    )
    evaluated = eclc(
        w_0=torch.rand(10),
        w_suffix=torch.rand(10),
        y_0=(torch.rand(10) > 0.5).float(),
        y_suffix=(torch.rand(10) > 0.5).float(),
    )
    assert len(evaluated) == 2
    assert "0" in evaluated
    assert "suffix" in evaluated


def get_ml_loss(loss_fct: Callable, td: MockData) -> dict[str, float]:
    return to_floats(
        loss_fct(
            x=td.x,
            particle_id=td.particle_id,
            reconstructable=td.reconstructable,
            pt=td.pt,
            eta=td.eta,
            batch=td.batch,
            true_edge_index=td.true_edge_index,
        ).loss_dct
    )


def test_hinge_loss():
    assert get_ml_loss(GraphConstructionHingeEmbeddingLoss(), td1) == approx(
        {"attractive": 0.7307405975481213, "repulsive": 0.34612957938781874}
    )


if __name__ == "__main__":
    for strategy in ["tiger", "rg"]:
        print(f"{strategy=}")
        print(get_condensation_loss(td1, strategy=strategy))
        print(get_condensation_loss(td2, strategy=strategy))
    for strategy in ["efficiency", "purity"]:
        print(f"{strategy=}")
        print(get_object_loss(td1, mode=strategy))
        print(get_object_loss(td2, mode=strategy))
    print(get_ml_loss(GraphConstructionHingeEmbeddingLoss(), td1))
