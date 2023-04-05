from __future__ import annotations

from functools import partial

import pytest

from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightBCELoss,
    LossFctType,
    PotentialLoss,
)
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.models.track_condensation_networks import (
    GraphTCN,
    PerfectECGraphTCN,
    PreTrainedECGraphTCN,
)
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.dynamiclossweights import NormalizeAt
from gnn_tracking.training.tcn_trainer import TCNTrainer, TrainingTruthCutConfig
from gnn_tracking.utils.seeds import fix_seeds

_test_train_test_cases = [
    ("graphtcn", "default"),
    ("graphtcn", "auto"),
    ("graphtcn", "default"),
    ("pretrainedec", "default"),
    ("perfectec", "default"),
]


@pytest.mark.parametrize("model,loss_weights", _test_train_test_cases)
def test_train(tmp_path, built_graphs, model: str, loss_weights: str) -> None:
    fix_seeds()
    _, graph_builder = built_graphs
    g = graph_builder.data_list[0]
    node_indim = g.x.shape[1]
    edge_indim = g.edge_attr.shape[1]

    graphs = graph_builder.data_list
    n_graphs = len(graphs)
    assert n_graphs == 2

    loaders = {
        "train": [graphs[0]],
        "test": [graphs[1]],
        "val": [graphs[1]],
    }

    q_min, sb = 0.01, 0.1
    loss_functions: dict[str, LossFctType] = {
        "edge": EdgeWeightBCELoss(),
        "potential": PotentialLoss(q_min=q_min),
        "background": BackgroundLoss(sb=sb),
    }

    # set up a model and trainer
    if model == "graphtcn":
        model_ = GraphTCN(
            node_indim,
            edge_indim,
            h_dim=2,
            hidden_dim=2,
            L_ec=2,
            L_hc=2,
        )
    elif model == "pretrainedec":
        ec = ECForGraphTCN(
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=2,
            L_ec=2,
        )
        model_ = PreTrainedECGraphTCN(
            ec,
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=2,
            L_hc=2,
        )
    elif model == "perfectec":
        model_ = PerfectECGraphTCN(
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=2,
            L_hc=2,
            ec_tpr=0.8,
            ec_tnr=0.4,
        )
    else:
        raise ValueError(f"Unknown model type {model}")

    _loss_weights = None
    if loss_weights == "default":
        _loss_weights = None
    elif loss_weights == "auto":
        _loss_weights = NormalizeAt(at=[0])
    else:
        raise ValueError()

    trainer = TCNTrainer(
        model=model_,
        loaders=loaders,
        loss_functions=loss_functions,
        lr=0.0001,
        cluster_functions={"dbscan": partial(dbscan_scan, n_trials=1)},  # type: ignore
        loss_weights=_loss_weights,
    )
    trainer.checkpoint_dir = tmp_path
    tcc = TrainingTruthCutConfig(
        pt_thld=0.01,
        without_noise=True,
        without_non_reconstructable=True,
    )
    trainer.training_truth_cuts = tcc

    trainer.train_step(max_batches=1)
    trainer.test_step(max_batches=1)
    trainer.save_checkpoint("model.pt")
    trainer.load_checkpoint("model.pt")
