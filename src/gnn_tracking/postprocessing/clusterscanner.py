from abc import ABC, abstractmethod
from typing import Any

from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor as T
from torch_geometric.data import Data


class ClusterScanner(
    HyperparametersMixin,
    ABC,
):
    def __init__(self, *args, **kwargs):
        """Base class for cluster scanners. Use any of its subclasses."""

        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(
        self,
        data: Data,
        out: dict[str, T],
        i_batch: int,
    ) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_foms(self) -> dict[str, Any]:
        return {}


class CombinedClusterScanner(ClusterScanner):
    def __init__(self, scanners: list[ClusterScanner]):
        """Combine multiple `ClusterScanner` objects."""
        super().__init__()
        self._scanners = scanners
        # todo: Hyperparameters aren't properly combined

    def __call__(self, *args, **kwargs):
        for scanner in self._scanners:
            scanner(*args, **kwargs)

    def reset(self) -> None:
        for scanner in self._scanners:
            scanner.reset()

    def get_foms(self) -> dict[str, Any]:
        foms = {}
        for scanner in self._scanners:
            foms |= scanner.get_foms()
        return foms
