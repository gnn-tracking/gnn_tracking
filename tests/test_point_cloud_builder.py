import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gnn_tracking.preprocessing.point_cloud_builder import (
    PointCloudBuilder,
    get_truth_edge_index,
    simple_data_loader,
)

from .test_data import trackml_test_data_dir


@pytest.fixture()
def test_data_files():
    hits, particles, truth, cells = simple_data_loader(
        trackml_test_data_dir / "event000000001"
    )
    return hits, particles, truth, cells


@pytest.fixture()
def point_cloud_builder():
    return PointCloudBuilder(
        outdir="tmp_output/",
        indir=trackml_test_data_dir,
        detector_config=trackml_test_data_dir / "detectors.csv.gz",
        n_sectors=1,
        redo=False,
        pixel_only=False,
        sector_di=0.0001,
        sector_ds=1.1,
        measurement_mode=False,
        thld=0.5,
        remove_noise=False,
        write_output=True,
        collect_data=True,
        add_true_edges=True,
    )


@pytest.fixture()
def point_cloud_builder_pixel():
    return PointCloudBuilder(
        outdir="tmp_output/pixel/",
        indir=trackml_test_data_dir,
        detector_config=trackml_test_data_dir / "detectors.csv.gz",
        n_sectors=1,
        redo=False,
        pixel_only=True,
        sector_di=0.0001,
        sector_ds=1.1,
        measurement_mode=False,
        thld=0.5,
        remove_noise=False,
        write_output=True,
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


def test_process_no_sectors(point_cloud_builder_pixel, test_data_files):
    point_cloud_builder_pixel.process(0, 1)
    f_path = Path("tmp_output/pixel/data1_s0.pt")
    graph_data = torch.load(f_path)
    f_path.unlink()
    original_hits, particles, truth, cells = test_data_files
    hits = original_hits.merge(truth, on="hit_id")
    hits = hits[hits["volume_id"].isin([7, 8, 9])]
    separate_check = ["x", "edge_index", "y"]
    length_check_keys = [
        key for key in graph_data.keys() if key not in separate_check  # noqa: SIM118
    ]
    for key in length_check_keys:
        assert len(graph_data[key]) == len(hits), (
            f"length of {key} "
            f"is {len(graph_data[key])} != len of hits is {len(hits)}"
        )

    assert graph_data.x.shape[0] == len(hits)
    assert graph_data.x.shape[1] == len(point_cloud_builder_pixel.feature_names)

    # on average 6 hits in a particle in pixel, expect fully connected edges between them
    expected_number_of_edges = hits.particle_id.nunique() * np.cumsum(range(6))[-1]
    actual_by_expected_edges = graph_data.edge_index.shape[1] / expected_number_of_edges

    assert graph_data.edge_index.shape[0] == 2
    assert 1.5 > actual_by_expected_edges > 0.5, (
        f"The number of edges seem off "
        f" expected {expected_number_of_edges}, got {graph_data.edge_index.shape[1]}"
    )


def test_process_sectors(point_cloud_builder, test_data_files):
    point_cloud_builder.process(0, 1)
    sector_graph_list = []
    for i in range(point_cloud_builder.n_sectors):
        sector_data = torch.load(f"tmp_output/data1_s{i}.pt")
        sector_graph_list.append(sector_data)

    graph_data = Batch.from_data_list(sector_graph_list)

    shutil.rmtree("tmp_output/")
    original_hits, particles, truth, cells = test_data_files
    hits = original_hits.merge(truth, on="hit_id")

    separate_check = ["x", "edge_index", "y", "ptr"]
    length_check_keys = [
        key for key in graph_data.keys() if key not in separate_check  # noqa: SIM118
    ]
    for key in length_check_keys:
        assert len(graph_data[key]) == len(hits), (
            f"length of {key} "
            f"is {len(graph_data[key])} != len of hits is {len(hits)}"
        )

    assert graph_data.x.shape[0] == len(hits)
    assert graph_data.x.shape[1] == len(point_cloud_builder.feature_names)

    # on average 6 hits in a particle in pixel, expect fully connected edges between them
    expected_number_of_edges = hits.particle_id.nunique() * np.cumsum(range(10))[-1]
    actual_by_expected_edges = graph_data.edge_index.shape[1] / expected_number_of_edges

    assert graph_data.edge_index.shape[0] == 2
    assert 1.5 > actual_by_expected_edges > 0.5, (
        f"The number of edges falls outside the expected range "
        f" expected {expected_number_of_edges}, got {graph_data.edge_index.shape[1]}"
    )
