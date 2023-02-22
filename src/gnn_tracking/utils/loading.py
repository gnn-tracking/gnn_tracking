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


def get_loaders(
    graph_dct: dict[str, list[Data]], *, batch_size=1, cpus=1
) -> dict[str, DataLoader]:
    """Get data loaders from a dictionary of lists of input graph.

    Args:
        graph_dct: Mapping from dataset name (e.g., train/test/val) to list of graphs
        batch_size: Batch size for data loaders
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
