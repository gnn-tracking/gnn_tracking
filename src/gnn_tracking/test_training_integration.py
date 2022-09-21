from __future__ import annotations

from functools import partial

from pytest import approx
from torch_geometric.loader import DataLoader

from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.losses import BackgroundLoss, EdgeWeightBCELoss, PotentialLoss
from gnn_tracking.utils.seeds import fix_seeds


def test_train(built_graphs):
    fix_seeds()
    _, graph_builder = built_graphs
    g = graph_builder.data_list[0]
    node_indim = g.x.shape[1]
    edge_indim = g.edge_attr.shape[1]

    graphs = graph_builder.data_list
    n_graphs = len(graphs)
    assert n_graphs == 2
    params = {"batch_size": 1, "shuffle": True, "num_workers": 1}
    train_graphs = DataLoader([graphs[0]], **params)
    test_graphs = DataLoader([graphs[1]], **params)
    val_graphs = DataLoader([graphs[1]], **params)

    train_loader = DataLoader(list(train_graphs), **params)

    params = {"batch_size": 2, "shuffle": False, "num_workers": 2}
    test_loader = DataLoader(list(test_graphs), **params)
    val_loader = DataLoader(list(val_graphs), **params)
    loaders = {"train": train_loader, "test": test_loader, "val": val_loader}
    print("Loader sizes:", [(k, len(v)) for k, v in loaders.items()])

    q_min, sb = 0.01, 0.1
    loss_functions = {
        "edge": EdgeWeightBCELoss().to(),
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
        cluster_functions={"dbscan": partial(dbscan_scan, n_trials=1)},
    )

    trainer.train(epochs=1, max_batches=1)
    result = trainer.test_step()
    assert result == approx(
        {
            "acc": 0.667654028436019,
            "TPR": 0.13602150537634408,
            "TNR": 0.8697874080130826,
            "FPR": 0.13021259198691743,
            "FNR": 0.863978494623656,
            "total": 46.57383346557617,
            "edge": 0.6902721524238586,
            "potential_attractive": 7.787227514199913e-05,
            "potential_repulsive": 42.47319030761719,
            "background": 0.6485034823417664,
            "v_measure": 0.0,
            "homogeneity": 0.0,
            "completeness": 1.0,
            "custom.total": 122.0,
            "custom.perfect": 0.0,
            "custom.double_majority": 0.0,
            "custom.lhc": 1.0,
            "adjusted_rand": 0.0,
            "fowlkes_mallows": 0.09654130313668478,
            "adjusted_mutual_info": 8.65042156952295e-17,
        }
    )
