import pytest

from gnn_tracking.metrics.cluster_metrics import (
    tracking_metrics_vs_eta,
    tracking_metrics_vs_pt,
)
from gnn_tracking.postprocessing.clusterscanner import (
    ClusterScanner,
    CombinedClusterScanner,
)
from gnn_tracking.postprocessing.dbscanscanner import (
    DBSCANHyperParamScanner,
    DBSCANHyperParamScannerFixed,
    DBSCANPerformanceDetails,
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
    scanner.get_foms()


def test_performance_details(test_graph):
    scanner = DBSCANPerformanceDetails(eps=0.5, min_samples=1)
    scanner(test_graph, {"H": test_graph.x}, 0)
    h_dfs, c_dfs = scanner.get_results()
    tracking_metrics_vs_pt(h_dfs, c_dfs, pts=[0.0, 0.5, 1.5], max_eta=4.0)
    tracking_metrics_vs_eta(h_dfs, c_dfs, pt_thld=0.5, etas=[0.0, 1.0, 4.0])
