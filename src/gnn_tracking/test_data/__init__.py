from __future__ import annotations

from pathlib import Path

from gnn_tracking.utils.compat_resource import resources

test_data_dir = Path(resources.files("gnn_tracking.test_data"))  # type: ignore
