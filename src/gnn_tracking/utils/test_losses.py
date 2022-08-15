from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from utils.losses import BackgroundLoss, ObjectLoss, PotentialLoss

T = torch.tensor


@dataclass
class TestData:
    beta: T
    x: T
    particle_id: T
    pred: T
    truth: T


def generate_test_data(
    n_nodes=1000, n_particles=250, n_x_features=3, rng=None
) -> TestData:
    if rng is None:
        rng = np.random.default_rng()
    return TestData(
        beta=torch.from_numpy(rng.random(n_nodes)),
        x=torch.from_numpy(rng.random((n_nodes, n_x_features))),
        particle_id=torch.from_numpy(rng.choice(np.arange(n_particles), size=n_nodes)),
        pred=torch.from_numpy(rng.choice([0.0, 1.0], size=(n_nodes, 1))),
        truth=torch.from_numpy(rng.choice([0.0, 1.0], size=(n_nodes, 1))),
    )


td1 = generate_test_data(10, n_particles=3, rng=np.random.default_rng(seed=0))
td2 = generate_test_data(20, n_particles=3, rng=np.random.default_rng(seed=0))


def get_condensation_loss(td: TestData) -> float:
    return PotentialLoss().condensation_loss(td.beta, td.x, td.particle_id).item()


def get_background_loss(td: TestData) -> float:
    return BackgroundLoss().background_loss(td.beta, td.particle_id)


def get_object_loss(td: TestData, **kwargs) -> float:
    return ObjectLoss(**kwargs).object_loss(
        beta=td.beta, particle_id=td.particle_id, pred=td.pred, truth=td.truth
    )


def test_potential_loss():
    assert np.isclose(get_condensation_loss(td1), 7.7166)
    assert np.isclose(get_condensation_loss(td2), 7.1898)


def test_background_loss():
    assert np.isclose(get_background_loss(td1).item(), 0.12870374134954846)
    assert np.isclose(get_background_loss(td2).item(), 0.16493608241281874)


def test_object_loss_efficiency():
    assert np.isclose(get_object_loss(td1).item(), 26.666667938232422)
    assert np.isclose(get_object_loss(td2).item(), 62.222225189208984)


def test_object_loss_purity():
    assert np.isclose(get_object_loss(td1, mode="purity").item(), 3.6432214679013644)
    assert np.isclose(get_object_loss(td2, mode="purity").item(), 3.8949206999806045)
