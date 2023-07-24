from torch_geometric.data import Data

from gnn_tracking.graph_construction.ml_graph_builder import MLGraphBuilder
from gnn_tracking.models.graph_construction import (
    GraphConstructionFCNN,
    MLGraphConstruction,
)
from tests.test_data import graph_test_data_dir


def test_ml_graph_builder(test_graph: Data, tmp_path):
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
    ml_graph_builder = MLGraphBuilder(
        gc=gc,
    )
    ml_graph_builder.process_directories(
        input_dirs=[graph_test_data_dir],
        output_dirs=[tmp_path / "transformed"],
        progress=False,
        _first_only=True,
    )
