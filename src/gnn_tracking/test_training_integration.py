from __future__ import annotations

from functools import partial

from torch_geometric.loader import DataLoader

from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.losses import BackgroundLoss, EdgeWeightBCELoss, PotentialLoss
from gnn_tracking.utils.seeds import fix_seeds


def test_train(tmp_path, built_graphs):
    fix_seeds()
    _, graph_builder = built_graphs
    g = graph_builder.data_list[0]
    node_indim = g.x.shape[1]
    edge_indim = g.edge_attr.shape[1]

    graphs = graph_builder.data_list
    n_graphs = len(graphs)
    assert n_graphs == 2
    params = {"batch_size": 1, "num_workers": 1}

    loaders = {
        "train": DataLoader([graphs[0]], **params, shuffle=True),
        "test": DataLoader([graphs[1]], **params),
        "val": DataLoader([graphs[1]], **params),
    }

    q_min, sb = 0.01, 0.1
    loss_functions = {
        "edge": EdgeWeightBCELoss(),
        "potential": PotentialLoss(q_min=q_min),
        "background": BackgroundLoss(sb=sb),
    }

    loss_weights = {
        "edge": 5,
        "potential_attractive": 10,
        "potential_repulsive": 1,
        "background": 1,
    }

    # set up a model and trainer
    model = GraphTCN(node_indim, edge_indim, h_dim=2, hidden_dim=64)

    trainer = TCNTrainer(
        model=model,
        loaders=loaders,
        loss_functions=loss_functions,
        lr=0.0001,
        loss_weights=loss_weights,
        cluster_functions={"dbscan": partial(dbscan_scan, n_trials=1)},  # type: ignore
    )
    trainer.checkpoint_dir = tmp_path

    trainer.train(epochs=1, max_batches=1)
    trainer.test_step()
    trainer.save_checkpoint("model.pt")
    trainer.load_checkpoint("model.pt")
