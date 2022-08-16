from __future__ import annotations

import logging
import os
from os.path import join

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def initialize_logger(verbose=False):
    log_format = "%(asctime)s %(levelname)s %(message)s"
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info("Initializing")


def get_graph_paths(indir):
    graph_files = os.listdir(indir)
    return np.array([join(indir, f) for f in graph_files])


def get_graph_evtids(graphs):
    return np.array([int(g.split("0000")[-1].split("_s")[0]) for g in graphs])


def sort_graph_paths(paths):
    evtids = np.array([int(g.split("0000")[-1].split("_s")[0]) for g in paths])
    unique_evtids = np.sort(np.unique(evtids))
    sorted_paths = []
    for evtid in unique_evtids:
        evtid_paths = paths[evtids == evtid]
        sectors = np.array([int(g.split("_s")[-1].split(".")[0]) for g in evtid_paths])
        sorted_sectors = np.argsort(sectors)
        sorted_paths.extend(list(evtid_paths[sorted_sectors]))
    return np.array(sorted_paths)


def clean_graph_paths(paths):
    empty_graphs = []
    for path in paths:
        f = np.load(path)
        if len(f["edge_index"][1]) == 0:
            empty_graphs.append(path)
    for g in empty_graphs:
        paths = paths[graph_paths != g]
    return paths


def partition_graphs(
    indir, n_train, n_test, n_val=0, shuffle=False, remove_empty_graphs=True
):
    paths = get_graph_paths(indir)
    if remove_empty_graphs:
        paths = clean_graph_paths(paths)
    n_graphs = len(paths)
    if (n_train + n_test + n_val) > n_graphs:
        print("The number of graphs requested exceeds number available.")
        return -1

    # optionally shuffle evt_ids
    evtids = np.arange(n_graphs)
    if shuffle:
        np.random.shuffle(evt_ids)
    train_ids = evtids[:n_train]
    n2 = n_train + n_test
    train_ids = evtids[:n_train]
    test_ids = evtids[n_train:n2]
    partition = {"train": paths[evtids[train_ids]], "test": paths[evtids[test_ids]]}

    # optionally add validation set
    if n_val > 0:
        val_ids = evtids[n2 : n2 + n_val]
        partition["val"] = paths[evtids[val_ids]]

    return partition


def get_graph_dataset(indir):
    paths = get_graph_paths(indir)
    return GraphDataset(graph_files=paths)


def get_dataloader(indir, params={}):
    dataset = get_graph_dataset(indir)
    return Dataloader(dataset, **params)


def get_dataloaders(
    indir, n_train, n_test, n_val=0, shuffle=False, params={}, remove_empty_graphs=False
):
    graph_parts = partition_graphs(
        indir,
        n_train,
        n_test,
        n_val=n_val,
        shuffle=shuffle,
        remove_empty_graphs=remove_empty_graphs,
    )
    loaders = {}
    for name, paths in graph_parts.items():
        dataset = GraphDataset(graph_files=paths)
        loaders[name] = DataLoader(dataset, **params)
    return loaders


class GraphDataset(Dataset):
    def __init__(
        self, transform=None, pre_transform=None, graph_files=[], bidirected=True
    ):
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_files = graph_files
        self.bidirected = bidirected

    @property
    def raw_file_names(self):
        return self.graph_files

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        with np.load(self.graph_files[idx]) as f:
            # load basic graph properties
            x = torch.from_numpy(f["x"])
            edge_attr = torch.from_numpy(f["edge_attr"])
            edge_index = torch.from_numpy(f["edge_index"])
            y = torch.from_numpy(f["y"])
            particle_id = torch.from_numpy(f["particle_id"])
            track_params = torch.from_numpy(f["track_params"])
            reconstructable = torch.from_numpy(f["reconstructable"])

            # some segmented graphs are empty
            if len(x) == 0:
                x = torch.tensor([], dtype=torch.float)
                particle_id = torch.tensor([], dtype=torch.long)
                edge_index = torch.tensor([[], []], dtype=torch.long)
                edge_attr = torch.tensor([], dtype=torch.float)
                y = torch.tensor([], dtype=torch.float)
                track_params = torch.tensor([[]], dtype=torch.float)
                reconstructable = torch.tensor([], dtype=torch.float)

            # make graph undirected
            if self.bidirected == True:
                row_0, col_0 = edge_index
                row = torch.cat([row_0, col_0], dim=0)
                col = torch.cat([col_0, row_0], dim=0)
                edge_index = torch.stack([row, col], dim=0)
                negate = torch.tensor([[-1], [-1], [-1], [1]])
                edge_attr = torch.cat([edge_attr, negate * edge_attr], dim=1)
                y = torch.cat([y, y])

            # return torch geometric data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=torch.transpose(edge_attr, 0, 1),
                y=y,
                particle_id=particle_id,
                track_params=track_params.float(),
                reconstructable=reconstructable.long(),
            )
            data.num_nodes = len(x)

        return (data, self.graph_files[idx])
