import numpy as np
import pytest

from gnn_tracking.preprocessing.point_cloud_builder import (
    PointCloudBuilder,
    get_truth_edge_index,
    simple_data_loader,
)


@pytest.fixture()
def test_data_files():
    hits, particles, truth, cells = simple_data_loader(
        "test_data/trackml/event000000001"
    )
    return hits, particles, truth, cells


@pytest.fixture()
def point_cloud_builder():
    return PointCloudBuilder(
        outdir="",
        indir="test_data/trackml/",
        detector_config="test_data/trackml/detectors.csv.gz",
        n_sectors=4,
        redo=False,
        pixel_only=False,
        sector_di=0.0001,
        sector_ds=1.1,
        measurement_mode=False,
        thld=0.5,
        remove_noise=False,
        write_output=False,
        collect_data=True,
        add_true_edges=True,
    )


@pytest.fixture()
def point_cloud_builder_pixel():
    return PointCloudBuilder(
        outdir="",
        indir="test_data/trackml/",
        detector_config="test_data/trackml/detectors.csv.gz",
        n_sectors=1,
        redo=False,
        pixel_only=True,
        sector_di=0.0001,
        sector_ds=1.1,
        measurement_mode=False,
        thld=0.5,
        remove_noise=False,
        write_output=False,
        collect_data=True,
        add_true_edges=True,
    )


ACCEPTABLE_RANGES = {
    "r": (0, 1026),  # Example range for 'r'
    "phi": (-np.pi, np.pi),  # Example range for 'phi'
    "z": (-3000, 3000),  # Example range for 'z'
    "u": (-1, 1),  # Example range for 'u'
    "v": (-1, 1),  # Example range for 'v'
    "charge_frac": (0, 1),  # Example range for 'charge_frac'
    "leta": (-5, 5),  # Example range for 'leta'
    "lphi": (-np.pi, np.pi),  # Example range for 'lphi'
    "lx": (-3000, 3000),  # Example range for 'lx'
    "ly": (-3000, 3000),  # Example range for 'ly'
    "lz": (-3000, 3000),  # Example range for 'lz'
    "geta": (-5, 5),  # Example range for 'geta'
    "gphi": (-np.pi, np.pi),  # Example range for 'gphi'
}


def test_append_features(point_cloud_builder, test_data_files):
    hits, particles, truth, cells = test_data_files
    updated_hits = point_cloud_builder.append_features(hits, particles, truth, cells)

    assert "r" in updated_hits.columns
    assert "phi" in updated_hits.columns
    assert "pt" in updated_hits.columns
    assert len(updated_hits) == len(hits)

    for feature, (min_val, max_val) in ACCEPTABLE_RANGES.items():
        assert (
            updated_hits[feature].between(min_val, max_val).all()
        ), f"{feature} is out of range"


def test_restrict_to_subdetectors_full_det(point_cloud_builder, test_data_files):
    hits, particles, truth, cells = test_data_files
    hits_new_layers, cells = point_cloud_builder.restrict_to_subdetectors(hits, cells)
    assert len(hits) == len(hits_new_layers), (
        f" full detector used, but when relabelling layer numbers, "
        f"the length changes: {len(hits)} != {len(hits_new_layers)}"
    )
    assert (
        len(hits[["volume_id", "layer_id", "layer"]].value_counts())
        == hits_new_layers["layer"].nunique()
    ), "the layer id remapping is not unique"


def test_restrict_to_subdetectors_pixel(point_cloud_builder_pixel, test_data_files):
    hits, particles, truth, cells = test_data_files

    hits_new_layers, cells = point_cloud_builder_pixel.restrict_to_subdetectors(
        hits, cells
    )
    hits_in_pixels = hits[hits["volume_id"].isin([7, 8, 9])]
    assert len(hits_in_pixels) == len(hits_new_layers), (
        f" when subsetting to pixels "
        f"the length changes: {len(hits_in_pixels)} != {len(hits_new_layers)}"
    )

    assert (
        len(hits_in_pixels[["volume_id", "layer_id", "layer"]].value_counts())
        == hits_new_layers["layer"].nunique()
    ), "the layer id remapping is not unique"


def test_point_cloud_builder(point_clouds_path):
    """Make sure that the fixture is being called"""
    assert point_clouds_path.is_dir()


def test_get_truth_edge_index():
    assert (
        get_truth_edge_index(np.array([0, 1, 2, 3, 2, 1, 0]))
        == np.array([[1, 2], [5, 4]])
    ).all()
