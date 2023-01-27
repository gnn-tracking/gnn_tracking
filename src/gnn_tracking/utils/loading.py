from __future__ import annotations

import sklearn.model_selection
from gnn_tracking_hpo.util.log import logger
from torch_geometric.data import Data, DataLoader


def train_test_val_split(
    data: list[Data], *, test_frac: float = 0.1, val_frac: float = 0.1
) -> dict[str, list[Data]]:
    """Split up data into train, test, and validation sets."""
    rest, test_graphs = sklearn.model_selection.train_test_split(
        data, test_size=test_frac
    )
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
    params = {
        "batch_size": batch_size,
        "num_workers": cpus,
    }
    logger.debug("Parameters for data loaders: %s", params)
    loaders = {
        "train": DataLoader(list(graph_dct["train"]), **params, shuffle=True),
        "test": DataLoader(list(graph_dct["test"]), **params),
        "val": DataLoader(list(graph_dct["val"]), **params),
    }
    return loaders
