from typing import Any

import coolname
import numpy as np
from rich.console import Console


def denote_pt(inpt, pt_min=0.0) -> Any:
    """Append suffix to designate pt threshold.
    If string is given, return string.
    If dict is given, modify all keys.
    """
    suffix = "" if np.isclose(pt_min, 0.0) else f"_pt{pt_min:.1f}"
    if isinstance(inpt, str):
        return f"{inpt}{suffix}"
    elif isinstance(inpt, dict):
        return {denote_pt(k, pt_min=pt_min): v for k, v in inpt.items()}
    msg = f"Cannot denote_pt for type {type(inpt)}."
    raise ValueError(msg)


def random_trial_name(print=True) -> str:
    """Generate a random trial name.

    Args:
        print: Whether to print the name
    """
    name = coolname.generate_slug(3)
    if print:
        c = Console(width=80)
        c.rule(f"[bold][yellow]{name}[/yellow][/bold]")
    return name
