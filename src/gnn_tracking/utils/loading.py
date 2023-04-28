from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import RandomSampler
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader

from gnn_tracking.utils.log import logger


# noinspection PyAbstractClass
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

    @staticmethod
    def _get_paths(
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


# noinspection PyAbstractClass
class InMemoryTrackingDataset(InMemoryDataset, TrackingDataset):
    pass


def get_loaders(
    ds_dct: dict[str, list[Data] | Dataset],
    *,
    batch_size=1,
    cpus=1,
    other_batch_size=1,
    max_sample_size: int | None = None,
) -> dict[str, DataLoader]:
    """Get data loaders from a dictionary of lists of input graph.

    Args:
        ds_dct: Mapping from dataset name (e.g., train/test/val) to list of graphs
            Special options apply to the 'train' key.
        batch_size: Batch size for training data loaders
        other_batch_size: Batch size for data loaders other than training
        cpus: Number of CPUs for data loaders
        max_sample_size: Maximum size of samples to load for 'train' per epoch.
            If None, all.
            This doesn't mean that the data loader will only load this many samples.
            Rather, this only affects the number of samples that are loaded per
            epoch.

    Returns:
        Dictionary of data loaders
    """

    def get_params(key: str) -> dict[str, Any]:
        sampler = None
        if key == "train" and len(ds_dct[key]):
            replacement = max_sample_size > len(ds_dct[key])
            sampler = RandomSampler(
                ds_dct[key],
                replacement=replacement,
                num_samples=max_sample_size,
            )
        return {
            "batch_size": batch_size if key == "train" else other_batch_size,
            "num_workers": max(1, min(len(ds_dct[key]), cpus)),
            "sampler": sampler,
            "pin_memory": True,
        }

    loaders = {}
    for key, ds in ds_dct.items():
        params = get_params(key)
        logger.debug("Parameters for data loader '%s': %s", key, params)
        loaders[key] = DataLoader(ds, **params)
    return loaders
