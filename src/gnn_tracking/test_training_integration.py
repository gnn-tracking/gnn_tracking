from dataclasses import dataclass
from typing import Any

import pytest
from pytorch_lightning import Trainer

from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.models.track_condensation_networks import (
    GraphTCN,
    PerfectECGraphTCN,
    PreTrainedECGraphTCN,
)
from gnn_tracking.training.tc import TCModule
from gnn_tracking.utils.loading import TestTrackingDataModule
from gnn_tracking.utils.log import logger
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
    dm = TestTrackingDataModule(graphs)

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
        msg = f"Unknown model type {t.model}"
        raise ValueError(msg)

    lmodel = TCModule(
        model=model,
    )
    logger.debug(lmodel.hparams)
    # Avoid testing with TPS
    trainer = Trainer(max_steps=1, accelerator="cpu")
    trainer.fit(lmodel, datamodule=dm)
