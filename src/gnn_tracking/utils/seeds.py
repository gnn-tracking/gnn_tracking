from __future__ import annotations

import random

import numpy as np
import torch


def fix_seeds() -> None:
    """Try to fix all random seeds for reproducible runs."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
