from pathlib import Path

import pytest

from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder

from .test_data import trackml_test_data_dir


@pytest.fixture(scope="session")
def point_clouds_path(tmp_path_factory) -> Path:
    out_path = Path(tmp_path_factory.mktemp("point_clouds"))
    pc_builder = PointCloudBuilder(
        indir=trackml_test_data_dir,
        outdir=str(out_path),
        n_sectors=2,
        pixel_only=True,
        redo=False,
        measurement_mode=False,
        thld=0.9,
        detector_config=trackml_test_data_dir / "detectors.csv.gz",
    )
    pc_builder.process()
    return out_path


@pytest.fixture(scope="session")
def built_graphs(point_clouds_path, tmp_path_factory) -> tuple[Path, GraphBuilder]:
    out_path = Path(tmp_path_factory.mktemp("graphs"))
    graph_builder = GraphBuilder(
        str(point_clouds_path),
        str(out_path),
        redo=False,
    )
    graph_builder.process(stop=None)
    return out_path, graph_builder
