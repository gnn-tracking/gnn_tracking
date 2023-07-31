import gnn_tracking
from gnn_tracking.utils.versioning import assert_version_geq


def test_versioning_geq():
    assert_version_geq(gnn_tracking.__version__)
