import itertools
import os
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import RandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from gnn_tracking.utils.log import logger


# noinspection PyAbstractClass
class TrackingDataset(Dataset):
    def __init__(
        self,
        in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop=None,
        sector: int | None = None,
    ):
        """Dataset for tracking applications

        Args:
            in_dir: Directory or list of directories containing the data files
            start: Index of the first file to be considered (with files from the
                in dirs considered in order)
            stop: Index of the last file to be considered
            sector: If not None, only files with this sector number will be considered
        """
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
                msg = f"Directory {d} does not exist."
                raise FileNotFoundError(msg)
        glob = "*.pt" if sector is None else f"*_s{sector}.pt"
        available_files = sorted(
            itertools.chain.from_iterable([Path(d).glob(glob) for d in in_dir])
        )

        if stop is not None and stop > len(available_files):
            # to avoid tracking wrong hyperparameters
            msg = (
                f"stop={stop} is larger than the number of files "
                "({len(available_files)})"
            )
            raise ValueError(msg)
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


class TrackingDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train: dict | None = None,
        val: dict | None = None,
        test: dict | None = None,
        cpus: int = 1,
    ):
        """This subclass of `LightningDataModule` configures all data for the
        ML pipeline.

        Args:
            train: Config dictionary for training data (see below)
            val: Config dictionary for validation data (see below)
            test: Config dictionary for test data (see below)
            cpus: Number of CPUs to use for loading data.


        The following keys are available for each config dictionary:

        - `dirs`: List of dirs to load from (required)
        - `start=0`: Index of first file to load
        - `stop=None`: Index of last file to load
        - `sector=None`: Sector to load from (if None, load all sectors)
        - `batch_size=1`: Batch size

        Training has the following additional keys:

        - `sample_size=None`: Number of samples to load for each epoch
            (if None, load all samples)
        """
        super().__init__()
        self._configs = {
            "train": self._fix_datatypes(train),
            "val": self._fix_datatypes(val),
            "test": self._fix_datatypes(test),
        }
        self._datasets = {}
        self._cpus = cpus

    @property
    def datasets(self) -> dict[str, TrackingDataset]:
        if not self._datasets:
            logger.error(
                "Datasets have not been loaded yet. Make sure to call the setup method."
            )
        return self._datasets

    @staticmethod
    def _fix_datatypes(dct: dict[str, Any] | None) -> dict[str, Any] | None:
        """Fix datatypes of config dictionary.
        This is necessary because when configuring values from the command line,
        all values might be strings.
        """
        if dct is None:
            return {}
        for key in ["start", "stop", "sector", "batch_size", "sample_size"]:
            if key in dct:
                dct[key] = int(dct[key])
        return dct

    def _get_dataset(self, key) -> TrackingDataset:
        config = self._configs[key]
        if not config:
            msg = f"DataLoaderConfig for key {key} is None."
            raise ValueError(msg)
        return TrackingDataset(
            in_dir=config["dirs"],
            start=config.get("start", 0),
            stop=config.get("stop", None),
            sector=config.get("sector", None),
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._datasets["train"] = self._get_dataset("train")
            self._datasets["val"] = self._get_dataset("val")
        elif stage == "test":
            self._datasets["test"] = self._get_dataset("test")
        else:
            _ = f"Unknown stage '{stage}'"
            raise ValueError(_)

    def _get_dataloader(self, key):
        sampler = None
        dataset = self._datasets[key]
        n_samples = len(dataset)
        if key == "train" and len(self._datasets[key]):
            if "max_sample_size" in self._configs[key]:
                msg = "max_sample_size has been replaced by sample_size"
                raise ValueError(msg)
            n_samples = self._configs[key].get("sample_size", len(dataset))
            sampler = RandomSampler(
                self._datasets[key],
                replacement=n_samples > len(dataset),
                num_samples=n_samples,
            )
        return DataLoader(
            dataset,
            batch_size=self._configs[key].get("batch_size", 1),
            num_workers=max(1, min(n_samples, self._cpus)),
            sampler=sampler,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")


class TestTrackingDataModule(LightningDataModule):
    def __init__(self, graphs: list[Data]):
        """Version of `TrackingDataLoader` only used for testing purposes."""
        super().__init__()
        self.graphs = graphs
        self.datasets = {
            "train": [graphs[0]],
            "val": [graphs[1]],
            "test": [graphs[1]],
        }

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(self.datasets["train"])

    def val_dataloader(self):
        return DataLoader(self.datasets["val"])

    def test_dataloader(self):
        return DataLoader(self.datasets["test"])
