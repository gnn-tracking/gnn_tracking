from __future__ import annotations

from pathlib import Path

import pytest

from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder
from gnn_tracking.test_data import trackml_test_data_dir


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
    )
    pc_builder.process()
    return out_path


def test_point_cloud_builder(point_clouds_path):
    """Make sure that the fixture is being called"""
    assert point_clouds_path.is_dir()
