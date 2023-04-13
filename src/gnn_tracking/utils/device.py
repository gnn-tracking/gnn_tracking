from __future__ import annotations

import os

import torch


def guess_device(inpt=None):
    """Setting the device (cpu/cuda) with the following fallback alternatives

    1. input
    2. environment variable GNN_TRACKING_DEVICE
    3. cuda if available, else cpu
    """
    if inpt:
        return inpt
    if inpt := os.environ.get("GNN_TRACKING_DEVICE"):
        return torch.device(inpt)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
