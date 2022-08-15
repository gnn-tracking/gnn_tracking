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


def get_condensation_loss(td: TestData) -> float:
    return PotentialLoss().condensation_loss(td.beta, td.x, td.particle_id).item()


def get_background_loss(td: TestData) -> float:
    return BackgroundLoss().background_loss(td.beta, td.particle_id)


def get_object_loss(td: TestData) -> float:
    return ObjectLoss().object_loss(
        beta=td.beta, particle_id=td.particle_id, pred=td.pred, truth=td.truth
    )


def test_potential_loss():
    td = generate_test_data(10, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(get_condensation_loss(td), 7.7166)
    td = generate_test_data(20, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(get_condensation_loss(td), 7.1898)


def test_background_loss():
    td = generate_test_data(10, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(
        get_background_loss(td).item(), 0.12870374134954846
    ), get_background_loss(td).item()
    td = generate_test_data(20, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(get_background_loss(td).item(), 0.16493608241281874)


def test_object_loss():
    td = generate_test_data(10, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(get_object_loss(td).item(), 26.666667938232422)
    td = generate_test_data(20, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(get_object_loss(td).item(), 62.222225189208984)
