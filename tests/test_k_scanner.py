from gnn_tracking.graph_construction.k_scanner import GraphConstructionKNNScanner


def test_k_scanner(test_graph):
    gscanner = GraphConstructionKNNScanner(ks=[1, 2], targets=[0.001])
    gscanner(test_graph, i_batch=0)
    gscanner(test_graph, i_batch=1)
    assert len(gscanner.results_raw) == 4
    gscanner(test_graph, i_batch=0)
    assert len(gscanner.results_raw) == 2
    gscanner(test_graph, i_batch=1)
    kscan_results = gscanner.get_results()
    assert "n_edges_frac_segment50_0" in kscan_results.get_foms()
