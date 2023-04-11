from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import sklearn.model_selection
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader

from gnn_tracking.utils.log import logger


# todo: Can we make this accept both fracs and absolute numbers?
def train_test_val_split(
    data: list[Data], *, test_frac: float = 0.1, val_frac: float = 0.1
) -> dict[str, list[Data]]:
    """Split up data into train, test, and validation sets."""
    assert 0 <= test_frac <= 1
    assert 0 <= val_frac <= 1
    if not data:
        return {"train": [], "val": [], "test": []}
    if test_frac + val_frac > 1:
        raise ValueError("test_frac and val_frac must sum to less than 1")
    rest, test_graphs = sklearn.model_selection.train_test_split(
        data, test_size=test_frac
    )
    if len(rest) == 0:
        # Avoid zero div error
        train_graphs = []
        val_graphs = []
    else:
        train_graphs, val_graphs = sklearn.model_selection.train_test_split(
            rest, test_size=val_frac / (1 - test_frac)
        )
    return {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }


class TrackingDataset(Dataset):
    """Dataloader that works with both point cloud and graphs."""

    def __init__(
        self,
        in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop=None,
        sector: int | None = None,
    ):
        super().__init__()
        self._processed_paths = self._get_paths(
            in_dir, start=start, stop=stop, sector=sector
        )

    def _get_paths(
        self,
        in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop: int | None = None,
        sector: int | None = None,
    ) -> list[Path]:
        """Collect all paths that should be in this dataset."""
        if start == stop:
            return []

        if not isinstance(in_dir, list):
            in_dir = [in_dir]
        for d in in_dir:
            if not Path(d).exists():
                raise FileNotFoundError(f"Directory {d} does not exist.")
        glob = "*.pt" if sector is None else f"*_s{sector}.pt"
        available_files = sorted(
            itertools.chain.from_iterable([Path(d).glob(glob) for d in in_dir])
        )

        if stop is not None and stop > len(available_files):
            # to avoid tracking wrong hyperparameters
            raise ValueError(
                f"stop={stop} is larger than the number of files "
                f"({len(available_files)})"
            )
        considered_files = available_files[start:stop]
        logger.info(
            "DataLoader will load %d graphs (out of %d available).",
            len(considered_files),
            len(available_files),
        )
        logger.debug(
            "First graph is %s, last graph is %s",
            considered_files[0],
            considered_files[-1],
        )
        return considered_files

    def len(self) -> int:
        return len(self._processed_paths)

    def get(self, idx: int) -> Data:
        return torch.load(self._processed_paths[idx])


class InMemoryTrackingDataset(InMemoryDataset, TrackingDataset):
    pass


def get_loaders(
    graph_dct: dict[str, list[Data] | Dataset],
    *,
    batch_size=1,
    cpus=1,
    other_batch_size=1,
) -> dict[str, DataLoader]:
    """Get data loaders from a dictionary of lists of input graph.

    Args:
        graph_dct: Mapping from dataset name (e.g., train/test/val) to list of graphs
        batch_size: Batch size for training data loaders
        other_batch_size: Batch size for data loaders other than training
        cpus: Number of CPUs for data loaders

    Returns:
        Dictionary of data loaders
    """

    def get_params(key: str) -> dict[str, Any]:
        shuffle = key == "train"
        if not graph_dct[key]:
            # Shuffle dataloader checks explicitly for empty list, so let's work
            # around that
            shuffle = False
        return {
            "batch_size": batch_size if key == "train" else other_batch_size,
            "num_workers": max(1, min(len(graph_dct[key]), cpus)),
            "shuffle": shuffle,
            "pin_memory": True,
        }

    loaders = {}
    for key, graphs in graph_dct.items():
        params = get_params(key)
        logger.debug("Parameters for data loader '%s': %s", key, params)
        loaders[key] = DataLoader(list(graphs), **params)
    return loaders
