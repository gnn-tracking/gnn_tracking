from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pytest import approx

from gnn_tracking.utils.losses import BackgroundLoss, ObjectLoss, PotentialLoss

T = torch.tensor


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
    return PotentialLoss().condensation_loss(td.beta, td.x, td.particle_id).item()


def get_background_loss(td: MockData) -> float:
    return BackgroundLoss().background_loss(td.beta, td.particle_id).item()


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
    assert get_object_loss(td1) == approx(29.833900451660156)
    assert get_object_loss(td2) == approx(77.24552154541016)


def test_object_loss_purity():
    assert get_object_loss(td1, mode="purity") == approx(3.6432214679013644)
    assert get_object_loss(td2, mode="purity") == approx(3.8949206999806045)
