import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from gnn_tracking.preprocessing.point_cloud_builder import (
    CMSPointCloudBuilder,
    TrackMLPointCloudBuilder,
)

from .test_data import cms_test_data_dir, trackml_test_data_dir


@pytest.fixture()
def test_data_files():
    hits, particles, truth, cells = TrackMLPointCloudBuilder.load_trackml_event(
        trackml_test_data_dir, "event000000001"
    )
    return hits, particles, truth, cells


@pytest.fixture()
def test_cms_mc_files(cms_point_cloud_builder):
    hits, cells = cms_point_cloud_builder.read_event(0)
    return hits, cells


@pytest.fixture()
def point_cloud_builder():
    return TrackMLPointCloudBuilder(
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
        data_type="TrackML",
    )


@pytest.fixture()
def point_cloud_builder_pixel():
    return TrackMLPointCloudBuilder(
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
        data_type="TrackML",
    )


@pytest.fixture()
def cms_point_cloud_builder():
    return CMSPointCloudBuilder(
        outdir="tmp_output/cms/",
        indir=cms_test_data_dir,
        detector_config=cms_test_data_dir / "cms_Extended2026D110_ModulePositions.csv",
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
        data_type="CMS",
    )


ACCEPTABLE_RANGES = {
    "r": (0, 1026),  # Example range for 'r'cl
    "phi": (-np.pi, np.pi),  # Example range for 'phi'
    "z": (-3000, 3000),  # Example range for 'z'
    "eta_rz": (-5, 5),
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

# cms uses cm, tml does not
acceptable_ranges_cms = ACCEPTABLE_RANGES.copy()
keys_to_divide = ["r", "z", "lx", "ly", "lz"]
CMS_ACCEPTABLE_RANGES = {
    k: (v[0] / 10, v[1] / 10) if k in keys_to_divide else v
    for k, v in acceptable_ranges_cms.items()
}
CMS_ACCEPTABLE_RANGES["r"] = (0, 130)


def test_read_event(point_cloud_builder, test_data_files):
    hits, particles, truth, cells = test_data_files
    updated_hits = point_cloud_builder.read_event(0, 1)

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
    hits_new_layers = point_cloud_builder.restrict_to_subdetectors(hits)
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

    hits_new_layers = point_cloud_builder_pixel.restrict_to_subdetectors(hits)
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
    builder = TrackMLPointCloudBuilder(
        outdir="tmp_output/",
        indir=trackml_test_data_dir,
        detector_config=trackml_test_data_dir / "detectors.csv.gz",
        n_sectors=1,
        data_type="TrackML",
    )
    assert (
        builder.get_truth_edge_index(np.array([0, 1, 2, 3, 2, 1, 0]))
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
    length_check_keys = [key for key in graph_data if key not in separate_check]
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
    length_check_keys = [key for key in graph_data if key not in separate_check]
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


def test_cms_assign_background_track_ids(cms_point_cloud_builder, test_cms_mc_files):
    hits, _ = test_cms_mc_files
    hits_w_bkg_pid = cms_point_cloud_builder.assign_background_track_ids(hits)
    assert len(hits) == len(hits_w_bkg_pid), "length of hits when "
    f"adding background track id goes from {hits} to {hits_w_bkg_pid}"
    background = hits_w_bkg_pid[hits_w_bkg_pid["particle_id"] < 0]
    signal = hits_w_bkg_pid[hits_w_bkg_pid["particle_id"] > 0]
    assert len(signal) / len(background) > 0.001, "ratio of "
    f"signal to background is {len(signal)} / {len(background)}"
    background_vc = background.value_counts(["particle_id"]).reset_index()
    max_hits_per_particle = max(background_vc["count"])
    num_track_hits_above_50 = len(background_vc[background_vc["count"] > 50])
    assert num_track_hits_above_50 < 200, "the number of tracks with more than"
    f"fifty hits is {num_track_hits_above_50}"
    assert max_hits_per_particle < 200, "the background track with most hits has "
    f"{max_hits_per_particle} hits"


def test_append_cell_features(cms_point_cloud_builder, test_cms_mc_files):
    hits, cells = test_cms_mc_files
    augmented_hits = cms_point_cloud_builder.append_cell_features(hits, cells)
    assert len(hits) == len(augmented_hits), "the number of hits when "
    f"appending cell features went from {len(hits)} to {len(augmented_hits)}"


def test_cms_processing(cms_point_cloud_builder):
    cms_point_cloud_builder.process(0, 1)
    f_path = Path("tmp_output/cms/part_0/data_0_s0.pt")
    graph_data = torch.load(f_path)
    # f_path.unlink()

    graph_data_features = graph_data.x
    graph_data_df = pd.DataFrame(
        graph_data_features, columns=CMS_ACCEPTABLE_RANGES.keys()
    )
    for feature, (min_val, max_val) in CMS_ACCEPTABLE_RANGES.items():
        assert (
            graph_data_df[feature].between(min_val, max_val).all()
        ), f"{feature} is out of range, {min_val, max_val}, {max(graph_data_df[feature])}"
