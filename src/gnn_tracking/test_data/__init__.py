from __future__ import annotations

from importlib import resources
from pathlib import Path

test_data_dir = Path(resources.files("gnn_tracking.test_data"))  # type: ignore

trackml_test_data_dir = test_data_dir / "trackml"
trackml_test_data_prefix = "event000000001"
