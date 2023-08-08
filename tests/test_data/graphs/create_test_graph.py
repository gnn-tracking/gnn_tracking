#!/usr/bin/env python3

"""Create a test graph by taking an existing graph and applying a very strict
pt cut.
"""

import sys

import torch

from gnn_tracking.utils.log import logger

if __name__ == "__main__":
    filename = sys.argv[1]
    data = torch.load(filename)
    if not hasattr(data, "eta"):
        logger.warning("Adding a fake eta, because not present in input file")
        data.eta = torch.full_like(data.particle_id, 3)
    if getattr(data, "edge_attr", None) is None:
        logger.warning("Adding a fake edge attr, because not present in input file")
        data.edge_attr = torch.rand((data.edge_index.shape[1], 14))
    data = data.subgraph(data.pt > 8)
    logger.info(f"Number of nodes: {data.num_nodes}")
    torch.save(data, "test_graph.pt")
