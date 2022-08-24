from __future__ import annotations

import torch


def assert_int(*args: torch.Tensor):
    """Asserts that all input tensors are of type int."""
    for arg in args:
        assert not torch.is_floating_point(arg), "Expected int tensor"
