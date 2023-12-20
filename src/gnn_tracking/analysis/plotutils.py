from __future__ import annotations

import os

from matplotlib import pyplot as plt


def add_watermark(ax: plt.Axes, txt: str = "") -> None:
    """Add a watermark to a plot"""
    if not txt:
        return
    ax.text(
        0.5,
        0.5,
        txt,
        transform=ax.transAxes,
        fontsize=40,
        color="gray",
        alpha=0.5,
        ha="center",
        va="center",
        rotation=30,
    )


class Plot:
    FILENAME = ""

    def __init__(self, *, ax: plt.Axes | None = None, watermark="", model=""):
        """Base class for certain plots. Provides watermark and 'save' functionality."""
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        add_watermark(self.ax, watermark)
        title = ""
        if model:
            title += f"{model=}"
        self.ax.set_title(title)

    def add_legend(self) -> None:
        self.ax.legend()

    def save(self, path: os.PathLike | str = "") -> None:
        if not path:
            path = self.FILENAME + ".pdf"
        self.ax.figure.savefig(str(path), bbox_inches="tight")
