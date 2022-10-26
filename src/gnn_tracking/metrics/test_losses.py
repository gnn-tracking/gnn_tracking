from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pytest import approx
from torch.nn.functional import binary_cross_entropy
from typing_extensions import TypeAlias

from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    ObjectLoss,
    PotentialLoss,
    binary_focal_loss,
)

T: TypeAlias = torch.tensor


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


def get_condensation_loss(td: MockData) -> float:
    loss_dct = PotentialLoss(q_min=0.01, radius_threshold=1)._condensation_loss(
        beta=td.beta,
        x=td.x,
        particle_id=td.particle_id,
        mask=torch.full((len(td.x),), True),
    )
    assert len(loss_dct) == 2
    return loss_dct["attractive"] + 10 * loss_dct["repulsive"]


def get_background_loss(td: MockData) -> float:
    return (
        BackgroundLoss(sb=0.1)
        ._background_loss(beta=td.beta, particle_id=td.particle_id)
        .item()
    )


def get_object_loss(td: MockData, **kwargs) -> float:
    return (
        ObjectLoss(**kwargs)
        .object_loss(
            beta=td.beta, particle_id=td.particle_id, pred=td.pred, truth=td.truth
        )
        .item()
    )


def test_potential_loss():
    assert get_condensation_loss(td1) == approx(7.716561306915411)
    assert get_condensation_loss(td2) == approx(7.189839086949652)


def test_background_loss():
    assert get_background_loss(td1) == approx(0.12870374134954846)
    assert get_background_loss(td2) == approx(0.16493608241281874)


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
        0.5 * binary_cross_entropy(inpt, target, reduction="mean")
    )
