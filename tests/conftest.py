from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder

from .test_data import graph_test_data_first, trackml_test_data_dir

# Settings
# --------


def by_slow_marker(item):
    """Get items by slow marker"""
    return 1 if item.get_closest_marker("slow") is not None else 0


def pytest_collection_modifyitems(items):
    """Move slow items last"""
    # https://stackoverflow.com/a/61539510/
    items.sort(key=by_slow_marker)


def pytest_addoption(parser):
    parser.addoption("--no-slow", action="store_true", help="skip slow tests")


def pytest_runtest_setup(item):
    """Allow to skip slow tests"""
    # https://stackoverflow.com/a/47567535/
    if "slow" in item.keywords and item.config.getoption("--no-slow"):
        pytest.skip("skipped if --no-slow option is used")


# Globally scoped fixtures
# -----------------------


@pytest.fixture(scope="session")
def point_clouds_path(tmp_path_factory) -> Path:
    out_path = Path(tmp_path_factory.mktemp("point_clouds"))
    pc_builder = PointCloudBuilder(
        indir=trackml_test_data_dir,
        outdir=str(out_path),
        n_sectors=2,
        pixel_only=True,
        redo=False,
        measurement_mode=True,
        thld=0.9,
        detector_config=trackml_test_data_dir / "detectors.csv.gz",
        add_true_edges=True,
    )
    pc_builder.process()
    return out_path


@pytest.fixture(scope="session")
def built_graphs(point_clouds_path, tmp_path_factory) -> tuple[Path, GraphBuilder]:
    out_path = Path(tmp_path_factory.mktemp("graphs"))
    graph_builder = GraphBuilder(
        str(point_clouds_path),
        str(out_path),
        redo=True,
        measurement_mode=True,
    )
    graph_builder.process(stop=None)
    return out_path, graph_builder


@pytest.fixture(scope="session")
def test_graph() -> Data:
    graph = torch.load(graph_test_data_first)
    # todo: This is a hack, but this is a feature with our new graphs built from
    #   point clouds with ML
    graph.true_edge_index = graph.edge_index
    return graph
