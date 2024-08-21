import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from gnn_tracking.preprocessing.point_cloud_builder import get_truth_edge_index
from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder
import pandas as pd
@pytest.fixture
def mock_input_data():
    # Fixture to provide mock data
    hits = pd.DataFrame({
        'hit_id': [1, 2, 3, 4],
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [0.5, 1.5, 2.5, 3.5],
        'z': [2.0, 4.0, 6.0, 8.0],
        'volume_id': [7, 8, 9, 7],
        'layer_id': [2, 4, 6, 8],
        'particle_id': [1, 2, 0, 10],
        'module_id': [1, 1, 3, 4],
    })
    particles = pd.DataFrame({
        'particle_id': [1, 10, 0, 2],
        'px': [1.0, 2.0, 3.0, 4.0],
        'py': [1.0, 1.5, 2.0, 3.5],
        'pz': [1.0, 3.0, 4.0, 5.0],
        'q': [1, -1, 1, -1],
        'vx': [0.0, 0.0, 0.0, 0.0],
        'vy': [0.0, 0.0, 0.0, 0.0],
    })
    truth = pd.DataFrame({
        'hit_id': [1, 2, 3, 4],
        'particle_id': [1, 2, 0, 10]
    })
    cells = pd.DataFrame({
        'hit_id': [1, 2, 3, 4],
        'ch0': [125, 313, 313, 302],
        'ch1': [1003, 805, 804, 549],
        'value': [0.3, 0.2, 0.1, 0.3]
    })
    return hits, particles, truth, cells


@pytest.fixture
def mock_directories(tmp_path):
    """Create mock directories and files for testing."""
    # Create mock input directory structure
    indir = tmp_path / "input"
    indir.mkdir()

    # Create mock output directory
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Create some mock CSV files as expected input
    hits_file = indir / "Mockevent000021119-hits.csv.gz"
    hits_file.write_text("hit_id,x,y,z\n1,0,0,0\n2,1,1,1")

    particles_file = indir / "Mockevent000021119-particles.csv.gz"
    particles_file.write_text("particle_id,px,py,pz\n1,0.1,0.1,0.1")

    truth_file = indir / "Mockevent000021119-truth.csv.gz"
    truth_file.write_text("hit_id,particle_id\n1,1\n2,1")

    cells_file = indir / "Mockevent000021119-cells.csv.gz"
    cells_file.write_text("hit_id,value\n1,0.1\n2,0.2")

    # Return the paths for use in the test
    return indir, outdir


@pytest.fixture
def point_cloud_builder(mock_directories):
    indir, outdir = mock_directories
    return PointCloudBuilder(
        outdir=outdir,
        indir=indir,
        detector_config="detector_kaggle.csv",
        n_sectors=2,
        redo=False,
        pixel_only=True,
        sector_di=0.0001,
        sector_ds=1.1,
        measurement_mode=False,
        thld=0.5,
        remove_noise=True,
        write_output=False,
        collect_data=True,
        add_true_edges=True,
    )

ACCEPTABLE_RANGES = {
    "r": (0, 1026),  # Example range for 'r'
    "phi": (-np.pi, np.pi),  # Example range for 'phi'
    "z": (-3000, 3000),  # Example range for 'z'
    "eta_rz": (-np.pi, np.pi),  # Example range for 'eta_rz'
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

def test_append_features(point_cloud_builder, mock_input_data):
    hits, particles, truth, cells = mock_input_data
    updated_hits = point_cloud_builder.append_features(hits, particles, truth, cells)

    assert 'r' in updated_hits.columns
    assert 'phi' in updated_hits.columns
    assert 'pt' in updated_hits.columns
    assert len(updated_hits) == len(hits)

    for feature, (min_val, max_val) in ACCEPTABLE_RANGES.items():
        assert updated_hits[feature].between(min_val, max_val).all(), f"{feature} is out of range"


def test_point_cloud_builder(point_clouds_path):
    """Make sure that the fixture is being called"""
    assert point_clouds_path.is_dir()


def test_get_truth_edge_index():
    assert (
        get_truth_edge_index(np.array([0, 1, 2, 3, 2, 1, 0]))
        == np.array([[1, 2], [5, 4]])
    ).all()



import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from gnn_tracking.preprocessing.point_cloud_builder import get_truth_edge_index
from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder
import pandas as pd
@pytest.fixture
def mock_input_data():
    # Fixture to provide mock data
    hits = pd.DataFrame({
        'hit_id': [1, 2, 3, 4],
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [0.5, 1.5, 2.5, 3.5],
        'z': [2.0, 4.0, 6.0, 8.0],
        'volume_id': [7, 8, 9, 7],
        'layer_id': [2, 4, 6, 8],
        'particle_id': [1, 2, 0, 10],
        'module_id': [1, 1, 3, 4],
    })
    particles = pd.DataFrame({
        'particle_id': [1, 10, 0, 2],
        'px': [1.0, 2.0, 3.0, 4.0],
        'py': [1.0, 1.5, 2.0, 3.5],
        'pz': [1.0, 3.0, 4.0, 5.0],
        'q': [1, -1, 1, -1],
        'vx': [0.0, 0.0, 0.0, 0.0],
        'vy': [0.0, 0.0, 0.0, 0.0],
    })
    truth = pd.DataFrame({
        'hit_id': [1, 2, 3, 4],
        'particle_id': [1, 2, 0, 10]
    })
    cells = pd.DataFrame({
        'hit_id': [1, 2, 3, 4],
        'ch0': [125, 313, 313, 302],
        'ch1': [1003, 805, 804, 549],
        'value': [0.3, 0.2, 0.1, 0.3]
    })
    return hits, particles, truth, cells


@pytest.fixture
def mock_directories(tmp_path):
    """Create mock directories and files for testing."""
    # Create mock input directory structure
    indir = tmp_path / "input"
    indir.mkdir()

    # Create mock output directory
    outdir = tmp_path / "output"
    outdir.mkdir()

    # Create some mock CSV files as expected input
    hits_file = indir / "Mockevent000021119-hits.csv.gz"
    hits_file.write_text("hit_id,x,y,z\n1,0,0,0\n2,1,1,1")

    particles_file = indir / "Mockevent000021119-particles.csv.gz"
    particles_file.write_text("particle_id,px,py,pz\n1,0.1,0.1,0.1")

    truth_file = indir / "Mockevent000021119-truth.csv.gz"
    truth_file.write_text("hit_id,particle_id\n1,1\n2,1")

    cells_file = indir / "Mockevent000021119-cells.csv.gz"
    cells_file.write_text("hit_id,value\n1,0.1\n2,0.2")

    # Return the paths for use in the test
    return indir, outdir


@pytest.fixture
def point_cloud_builder(mock_directories):
    indir, outdir = mock_directories
    return PointCloudBuilder(
        outdir=outdir,
        indir=indir,
        detector_config="detector_kaggle.csv",
        n_sectors=2,
        redo=False,
        pixel_only=True,
        sector_di=0.0001,
        sector_ds=1.1,
        measurement_mode=False,
        thld=0.5,
        remove_noise=True,
        write_output=False,
        collect_data=True,
        add_true_edges=True,
    )

ACCEPTABLE_RANGES = {
    "r": (0, 1026),  # Example range for 'r'
    "phi": (-np.pi, np.pi),  # Example range for 'phi'
    "z": (-3000, 3000),  # Example range for 'z'
    "eta_rz": (-np.pi, np.pi),  # Example range for 'eta_rz'
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

def test_append_features(point_cloud_builder, mock_input_data):
    hits, particles, truth, cells = mock_input_data
    updated_hits = point_cloud_builder.append_features(hits, particles, truth, cells)

    assert 'r' in updated_hits.columns
    assert 'phi' in updated_hits.columns
    assert 'pt' in updated_hits.columns
    assert len(updated_hits) == len(hits)

    for feature, (min_val, max_val) in ACCEPTABLE_RANGES.items():
        assert updated_hits[feature].between(min_val, max_val).all(), f"{feature} is out of range"


def test_point_cloud_builder(point_clouds_path):
    """Make sure that the fixture is being called"""
    assert point_clouds_path.is_dir()


def test_get_truth_edge_index():
    assert (
        get_truth_edge_index(np.array([0, 1, 2, 3, 2, 1, 0]))
        == np.array([[1, 2], [5, 4]])
    ).all()



