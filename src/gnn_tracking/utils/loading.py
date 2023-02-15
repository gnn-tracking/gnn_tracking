from __future__ import annotations

from typing import Any

import sklearn.model_selection
from gnn_tracking_hpo.util.log import logger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def train_test_val_split(
    data: list[Data], *, test_frac: float = 0.1, val_frac: float = 0.1
) -> dict[str, list[Data]]:
    """Split up data into train, test, and validation sets."""
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


def get_loaders(
    graph_dct: dict[str, list[Data]], *, batch_size=1, cpus=1
) -> dict[str, DataLoader]:
    """Get data loaders

    Args:
        graph_dct:
        batch_size:
        cpus: Number of CPUs for data loaders

    Returns:
        Dictionary of data loaders
    """

    def get_params(key: str) -> dict[str, Any]:
        return {
            "batch_size": batch_size,
            "num_workers": cpus,
            "shuffle": key == "train",
            "pin_memory": True,
        }

    loaders = {}
    for key, graphs in graph_dct.items():
        params = get_params(key)
        logger.debug("Parameters for data loader '%s': %s", key, params)
        loaders[key] = DataLoader(list(graphs), **params)
    return loaders
