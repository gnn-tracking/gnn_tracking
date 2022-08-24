from __future__ import annotations

from pathlib import Path

import pytest

from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder
from gnn_tracking.test_data import trackml_test_data_dir


@pytest.fixture(scope="session")
def point_cloud_builder(tmp_path_factory):
    pc_builder = PointCloudBuilder(
        indir=trackml_test_data_dir,
        outdir=str(tmp_path_factory.mktemp("point_clouds")),
        n_sectors=2,
        pixel_only=True,
        redo=False,
        measurement_mode=False,
        thld=0.9,
    )
    pc_builder.process(verbose=True)


def test_point_cloud_builder(point_cloud_builder):
    pass
