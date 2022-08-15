from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from utils.losses import PotentialLoss

T = torch.tensor


@dataclass
class TestData:
    beta: T
    x: T
    particle_id: T


def generate_test_data(
    n_nodes=1000, n_particles=250, n_x_features=3, rng=None
) -> TestData:
    if rng is None:
        rng = np.random.default_rng()
    return TestData(
        beta=torch.from_numpy(rng.random(n_nodes)),
        x=torch.from_numpy(rng.random((n_nodes, n_x_features))),
        particle_id=torch.from_numpy(rng.choice(np.arange(n_particles), size=n_nodes)),
    )


def get_condensation_loss(td: TestData) -> float:
    return PotentialLoss().condensation_loss(td.beta, td.x, td.particle_id).item()


def test_potential_loss():
    td = generate_test_data(10, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(get_condensation_loss(td), 7.7166)
    td = generate_test_data(20, n_particles=3, rng=np.random.default_rng(seed=0))
    assert np.isclose(get_condensation_loss(td), 7.1898)
