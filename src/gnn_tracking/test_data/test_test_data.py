from gnn_tracking.test_data import (
    test_data_dir,
    trackml_test_data_dir,
    trackml_test_data_prefix,
)


def test_test_data_dir():
    assert test_data_dir.is_dir()
    assert trackml_test_data_dir.is_dir()
    assert list(trackml_test_data_dir.glob(f"{trackml_test_data_prefix}*"))
