from __future__ import annotations


def zero_division_gives_nan(a: float, b: float) -> float:
    """Divide and return nan if we divide by zero"""
    try:
        return a / b
    except ZeroDivisionError:
        return float("nan")
