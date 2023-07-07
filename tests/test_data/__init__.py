from pathlib import Path

test_data_dir = Path(__file__).resolve().parent

trackml_test_data_dir = test_data_dir / "trackml"
trackml_test_data_prefix = "event000000001"
trackml_test_data_detector_config = trackml_test_data_dir / "detectors.csv.gz"
