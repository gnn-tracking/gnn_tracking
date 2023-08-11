import pytest

from gnn_tracking.postprocessing.clusterscanner import (
    ClusterScanner,
    CombinedClusterScanner,
)
from gnn_tracking.postprocessing.dbscanscanner import (
    DBSCANHyperParamScanner,
    DBSCANHyperParamScannerFixed,
)


def test_combined_cluster_scanner_empty():
    scanner = CombinedClusterScanner([])
    scanner()
    scanner.reset()
    assert scanner.get_foms() == {}


_scanners = [
    DBSCANHyperParamScanner(n_trials=2, keep_best=1),
    DBSCANHyperParamScannerFixed(trials=[{"eps": 0.5, "min_samples": 1}]),
]


@pytest.mark.parametrize("scanner", _scanners)
def test_clusterscanner(test_graph, scanner: ClusterScanner):
    scanner(test_graph, {"H": test_graph.x}, 0)
    assert "trk.double_majority_pt0.9" in scanner.get_foms()
