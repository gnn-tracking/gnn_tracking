from __future__ import annotations

import os
import sys

from point_cloud_builder import PointCloudBuilder

sys.path.append("../")

# configure initial params
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
indir = "/tigress/jdezoort/codalab/train_1"
outdir = "/tigress/jdezoort/object_condensation/point_clouds"

pc_builder = PointCloudBuilder(
    indir=indir,
    outdir=outdir,
    n_sectors=32,
    pixel_only=True,
    redo=True,
    measurement_mode=False,
    sector_di=0,
    sector_ds=1.3,
    thld=0.9,
    log_level=0,
)
pc_builder.process(start=idx, stop=idx + 1)
