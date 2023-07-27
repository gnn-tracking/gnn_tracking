from pathlib import Path

test_data_dir = Path(__file__).resolve().parent
assert test_data_dir.is_dir()
trackml_test_data_dir = test_data_dir / "trackml"
assert trackml_test_data_dir.is_dir()
trackml_test_data_prefix = "event000000001"
trackml_test_data_detector_config = trackml_test_data_dir / "detectors.csv.gz"
assert trackml_test_data_detector_config.is_file()
graph_test_data_dir = test_data_dir / "graphs"
assert graph_test_data_dir.is_dir()
graph_test_data_first = graph_test_data_dir / "test_graph.pt"
assert graph_test_data_first.is_file()
