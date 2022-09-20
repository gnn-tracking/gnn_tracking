from __future__ import annotations


def test_point_cloud_builder(point_clouds_path):
    """Make sure that the fixture is being called"""
    assert point_clouds_path.is_dir()
