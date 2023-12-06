from dataclasses import dataclass

import numpy as np
import torch
from pytest import approx  # noqa: PT013
from torch.nn.functional import binary_cross_entropy
from typing_extensions import TypeAlias

from gnn_tracking.metrics.losses import (
    LossClones,
    unpack_loss_returns,
)
from gnn_tracking.metrics.losses.ec import EdgeWeightBCELoss, binary_focal_loss
from gnn_tracking.metrics.losses.oc import (
    CondensationLossRG,
    CondensationLossTiger,
    ObjectLoss,
)

T: TypeAlias = torch.Tensor


@dataclass
class MockData:
    beta: T
    x: T
    particle_id: T
    pred: T
    truth: T


def generate_test_data(
    n_nodes=1000, n_particles=250, n_x_features=3, rng=None
) -> MockData:
    if rng is None:
        rng = np.random.default_rng()
    return MockData(
        beta=torch.from_numpy(rng.random(n_nodes)),
        x=torch.from_numpy(rng.random((n_nodes, n_x_features))),
        particle_id=torch.from_numpy(rng.choice(np.arange(n_particles), size=n_nodes)),
        pred=torch.from_numpy(rng.choice([0.0, 1.0], size=(n_nodes, 1))),
        truth=torch.from_numpy(rng.choice([0.0, 1.0], size=(n_nodes, 1))),
    )


td1 = generate_test_data(10, n_particles=3, rng=np.random.default_rng(seed=0))
td2 = generate_test_data(20, n_particles=3, rng=np.random.default_rng(seed=0))


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
    loss_dct = loss_fct(
        beta=td.beta,
        x=td.x,
        particle_id=td.particle_id,
        reconstructable=torch.full((len(td.x),), True),
        pt=torch.full((len(td.x),), 2),
        eta=torch.full((len(td.x),), 2.0),
    ).loss_dct
    assert len(loss_dct) > 2
    return loss_dct["attractive"] + 10 * loss_dct["repulsive"]


def get_object_loss(td: MockData, **kwargs) -> float:
    return (
        ObjectLoss(**kwargs)
        .object_loss(
            beta=td.beta, particle_id=td.particle_id, pred=td.pred, truth=td.truth
        )
        .item()
    )


def test_potential_loss():
    assert get_condensation_loss(td1) == approx(6.459650814283677)
    assert get_condensation_loss(td2) == approx(5.636204987639555)


def test_object_loss_efficiency():
    assert get_object_loss(td1) == approx(0.29833901913542193)
    assert get_object_loss(td2) == approx(0.7724552036470809)


def test_object_loss_purity():
    assert get_object_loss(td1, mode="purity") == approx(0.03643221467901364)
    assert get_object_loss(td2, mode="purity") == approx(0.038949206999806044)


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


def test_unpack_loss_returns():
    assert unpack_loss_returns("a", 2) == {"a": 2}
    assert unpack_loss_returns("a", {"b": 2}) == {"a_b": 2}
    assert unpack_loss_returns("a", [2]) == {"a_0": 2}
    assert unpack_loss_returns("a", []) == {}


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
