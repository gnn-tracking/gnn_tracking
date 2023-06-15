import itertools
import os
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import RandomSampler
from torch_geometric.data import Data, Dataset
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


class TrackingDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train: dict | None = None,
        val: dict | None = None,
        test: dict | None = None,
        cpus: int = 1,
    ):
        """

        Args:
            train:
            val:
            test:
            cpus:


        The following keys are available for each config dictionary:

        - `dirs`: List of dirs to load from (required)
        - `start=0`: Index of first file to load
        - `stop=None`: Index of last file to load
        - `sector=None`: Sector to load from (if None, load all sectors)
        - `batch_size=1`: Batch size

        Training has the following additional keys:

        - `max_sample_size=None`: Maximum number of samples to load for each epoch
            (if None, load all samples)
        """
        super().__init__()
        self._configs = {
            "train": train,
            "val": val,
            "test": test,
        }
        self._datasets = {}
        self._cpus = cpus

    def _get_dataset(self, key) -> TrackingDataset:
        config = self._configs[key]
        if config is None:
            raise ValueError(f"DataLoaderConfig for key {key} is None.")
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
            max_sample_size = self._configs[key].get("max_sample_size", None)
            n_samples = (
                min(n_samples, max_sample_size) if max_sample_size else n_samples
            )
            replacement = (max_sample_size > len(dataset)) if max_sample_size else False
            sampler = RandomSampler(
                self._datasets[key],
                replacement=replacement,
                num_samples=max_sample_size,
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
