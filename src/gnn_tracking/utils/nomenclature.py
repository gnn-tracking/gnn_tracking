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
    if isinstance(inpt, dict):
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


class Variable:
    def __init__(self, name: str, latex: str = ""):
        """Variable with latex expressions etc. To be used
        for VariableManager
        """
        self.name = name
        self._latex = latex

    @property
    def latex(self) -> str:
        return self._latex or self.name

    def __str__(self) -> str:
        return self.name


class VariableManager:
    def __init__(self):
        """Keep track of variables and their latex expressions etc."""
        self._variables: dict[str, Variable] = {}

    def __getitem__(self, name: str) -> Variable:
        try:
            return self._variables[name]
        except KeyError:
            return Variable(name)

    def add(self, other: list | Variable | tuple) -> None:
        """Add more variables to the manager"""
        if isinstance(other, list):
            for var in other:
                self.add(var)
        elif isinstance(other, Variable):
            self._variables[other.name] = other
        elif isinstance(other, tuple):
            self._variables[other[0]] = Variable(*other)
        else:
            msg = f"Cannot add {other} of type {type(other)}"
            raise TypeError(msg)


#: Pre-configured variable manager
variable_manager = VariableManager()
variable_manager.add(
    [
        ("frac50", "50SF"),
        ("frac75", "75SF"),
        ("frac100", "100SF"),
        ("efficiency", "Efficiency"),
        ("purity", "Purity"),
        ("double_majority", r"$\epsilon^{\mathrm{DM}}$"),
        ("double_majority_pt0.9", r"$\epsilon^{\mathrm{DM}}_{p_T > 0.9}$"),
        ("lhc", r"$\epsilon^{\mathrm{LHC}}$"),
        ("lhc_pt0.9", r"$\epsilon^{\mathrm{LHC}}_{p_T > 0.9}$"),
        ("perfect", r"$\epsilon^{\mathrm{perfect}}$"),
        ("perfect_pt0.9", r"$\epsilon^{\mathrm{perfect}}_{p_T > 0.9}$"),
        ("pt", "$p_T$"),
        ("eta", r"$\eta$"),
    ]
)
for target in ["90", "93", "95", "97"]:
    variable_manager.add(
        (
            f"n_edges_frac_segment50_{target}",
            r"$N_\text{edges}^{50\mathrm{SF}\geq " + target + r"\%}$",
        )
    )
