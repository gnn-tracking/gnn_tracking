from __future__ import annotations

import os
import sys

from graph_builder import GraphBuilder

sys.path.append("../")

# configure initial params
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
indir = "/tigress/jdezoort/object_condensation/point_clouds"
outdir = "/tigress/jdezoort/object_condensation/graphs"

graph_builder = GraphBuilder(
    indir="../point_clouds/for_paper",
    outdir="../graphs/for_paper",
    redo=False,
    measurement_mode=False,
    phi_slope_max=0.004,
    z0_max=225,
    dR_max=2.5,
    log_level=0,
)
graph_builder.process(start=idx, stop=idx + 1)
