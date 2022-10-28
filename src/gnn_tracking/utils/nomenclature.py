from __future__ import annotations

import numpy as np


def denote_pt(name: str, pt_min=0.0) -> str:
    """Suffix to append to designate pt threshold"""
    if np.isclose(pt_min, 0.0):
        return name
    return f"{name}_pt{pt_min:.1f}"
