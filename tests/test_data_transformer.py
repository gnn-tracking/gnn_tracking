from torch_geometric.data import Data

from gnn_tracking.graph_construction.data_transformer import DataTransformer, ECCut
from gnn_tracking.models.edge_filter import EFMLP
from gnn_tracking.models.graph_construction import (
    GraphConstructionFCNN,
    MLGraphConstruction,
)
from tests.test_data import graph_test_data_dir


def test_data_transformer(test_graph: Data, tmp_path):
    ml = GraphConstructionFCNN(
        in_dim=test_graph.num_node_features,
        out_dim=test_graph.num_node_features,
        hidden_dim=3,
        depth=1,
    )
    gc = MLGraphConstruction(
        ml=ml,
        use_embedding_features=True,
    )
    ml_graph_builder = DataTransformer(
        transform=gc,
    )
    ml_graph_builder.process_directories(
        input_dirs=[graph_test_data_dir],
        output_dirs=[tmp_path / "transformed"],
        _first_only=True,
    )


def test_eccut(test_graph: Data, tmp_path):
    ec = EFMLP(
        node_indim=test_graph.num_node_features,
        edge_indim=test_graph.num_edge_features,
        hidden_dim=3,
        depth=1,
    )
    dt = DataTransformer(
        transform=ECCut(ec, thld=0.3),
    )
    dt.process_directories(
        input_dirs=[graph_test_data_dir],
        output_dirs=[tmp_path / "transformed"],
        _first_only=True,
    )
