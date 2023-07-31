from pathlib import Path

__all__ = ["__version__"]

__version__ = (Path(__file__).resolve().parent / "version.txt").read_text().strip()
