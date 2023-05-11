from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

from gnn_tracking.metrics.losses import BackgroundLoss, EdgeWeightBCELoss, PotentialLoss
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.models.track_condensation_networks import (
    GraphTCN,
    PerfectECGraphTCN,
    PreTrainedECGraphTCN,
)
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.seeds import fix_seeds


@dataclass
class TestTrainCase:
    model: str = "graphtcn"
    loss_weights: str = "default"
    ec_params: dict[str, Any] | None = None
    tc_params: dict[str, Any] | None = None

    def __post_init__(self):
        if self.ec_params is None:
            self.ec_params = {}
        if self.tc_params is None:
            self.tc_params = {}


_test_train_test_cases = [
    TestTrainCase(
        "graphtcn",
    ),
    TestTrainCase("graphtcn", loss_weights="auto"),
    TestTrainCase(
        "graphtcn",
    ),
    TestTrainCase(
        "graphtcn",
        tc_params={"mask_orphan_nodes": True},
    ),
    TestTrainCase(
        "graphtcn",
        tc_params={"mask_orphan_nodes": True, "use_ec_embeddings_for_hc": True},
    ),
    TestTrainCase(
        "pretrainedec",
    ),
    TestTrainCase(
        "pretrainedec",
        ec_params={"residual_type": "skip2"},
    ),
    TestTrainCase(
        "pretrainedec",
        ec_params={"residual_type": "skip_top"},
    ),
    TestTrainCase(
        "pretrainedec",
        ec_params={"use_intermediate_edge_embeddings": False},
    ),
    TestTrainCase(
        "pretrainedec",
        ec_params={
            "use_intermediate_edge_embeddings": False,
            "use_node_embedding": False,
        },
    ),
    TestTrainCase(
        "pretrainedec",
        ec_params={"use_node_embedding": False},
    ),
    TestTrainCase(
        "pretrainedec",
        tc_params={"use_ec_embeddings_for_hc": True},
    ),
    TestTrainCase(
        "perfectec",
    ),
]


@pytest.mark.parametrize("t", _test_train_test_cases)
def test_train(tmp_path, built_graphs, t: TestTrainCase) -> None:
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
    loss_functions = {
        "edge": (EdgeWeightBCELoss(), 1.0),
        "potential": (
            PotentialLoss(q_min=q_min),
            {"attractive": 1.0, "repulsive": 1.0},
        ),
        "background": (BackgroundLoss(sb=sb), 1.0),
    }

    # set up a model and trainer
    if t.model == "graphtcn":
        model = GraphTCN(
            node_indim,
            edge_indim,
            h_dim=2,
            hidden_dim=2,
            L_ec=2,
            L_hc=2,
            **t.tc_params,
        )
    elif t.model == "pretrainedec":
        ec = ECForGraphTCN(
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=2,
            L_ec=2,
            **t.ec_params,
        )
        model = PreTrainedECGraphTCN(
            ec,
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=2,
            L_hc=2,
            **t.tc_params,
        )
    elif t.model == "perfectec":
        model = PerfectECGraphTCN(
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=2,
            L_hc=2,
            ec_tpr=0.8,
            ec_tnr=0.4,
            **t.tc_params,
        )
    else:
        raise ValueError(f"Unknown model type {t.model}")

    trainer = TCNTrainer(
        model=model,
        loaders=loaders,
        loss_functions=loss_functions,
        lr=0.0001,
        cluster_functions={"dbscan": partial(dbscan_scan, n_trials=1)},  # type: ignore
    )
    trainer.checkpoint_dir = tmp_path

    trainer.step(max_batches=1)
    trainer.save_checkpoint("model.pt")
    trainer.load_checkpoint("model.pt")
