from __future__ import annotations

import torch


def guess_device(inpt=None):
    """Return specified device or guess it"""
    if inpt is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return inpt
