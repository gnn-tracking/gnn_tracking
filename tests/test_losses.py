from dataclasses import dataclass

import numpy as np
import torch
from pytest import approx  # noqa: PT013
from torch.nn.functional import binary_cross_entropy
from typing_extensions import TypeAlias

from gnn_tracking.metrics.losses import (
    LossClones,
)
from gnn_tracking.metrics.losses.ec import EdgeWeightBCELoss, binary_focal_loss
from gnn_tracking.metrics.losses.oc import (
    CondensationLossRG,
    CondensationLossTiger,
    ObjectLoss,
)
from gnn_tracking.utils.dictionaries import to_floats

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
    return to_floats(
        loss_fct(
            beta=td.beta,
            x=td.x,
            particle_id=td.particle_id,
            reconstructable=torch.full((len(td.x),), True),
            pt=torch.full((len(td.x),), 2),
            eta=torch.full((len(td.x),), 2.0),
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


def test_pin_condensation_losses_tiger():
    assert get_condensation_loss(td1) == approx(
        {
            "attractive": 0.8338060530831202,
            "repulsive": 0.5625844761200557,
            "coward": 0.12582866850597973,
            "noise": 0.02875057973236189,
        }
    )
    assert get_condensation_loss(td2) == approx(
        {
            "attractive": 0.5407472117522135,
            "repulsive": 0.5095457775887342,
            "coward": 0.10376164981233127,
            "noise": 0.6117443924150829,
        }
    )


def test_pin_condensation_losses_rg():
    assert get_condensation_loss(td1, strategy="rg") == approx(
        {
            "attractive": 1.005005335056053,
            "repulsive": 0.6912557021450921,
            "coward": 0.12582866850597968,
            "noise": 0.02875057973236189,
        }
    )
    assert get_condensation_loss(td2, strategy="rg") == approx(
        {
            "attractive": 0.42965914163139446,
            "repulsive": 0.6889077680825523,
            "coward": 0.10376164981233121,
            "noise": 0.611744392415083,
        }
    )


def test_pin_object_loss_efficiency():
    assert get_object_loss(td1) == approx(0.29833901913542193)
    assert get_object_loss(td2) == approx(0.7724552036470809)


def test_pin_object_loss_purity():
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


if __name__ == "__main__":
    for strategy in ["tiger", "rg"]:
        print(f"{strategy=}")  # noqa: T201
        print(get_condensation_loss(td1, strategy=strategy))  # noqa: T201
        print(get_condensation_loss(td2, strategy=strategy))  # noqa: T201
