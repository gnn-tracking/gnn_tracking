from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from graph_builder import GraphBuilder

from gnn_tracking.utils.log import logger
from gnn_tracking.utils.versioning import get_commit_hash

N_GRAPHS_MEASURED = 10
N_EXPERIMENTS = 1
GRAPH_INDIR = Path(
    "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v2/part_1"
)
MEASUREMENT_OUTDIR = Path(
    "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/measurements"
)

if __name__ == "__main__":
    results = {
        "gnn_tracking_commit_hash": get_commit_hash(),
        "date": datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
        "results": [],
        "input_dir": str(GRAPH_INDIR),
        "n_graphs_measured": N_GRAPHS_MEASURED,
    }
    logger.debug(results)
    for _ in range(N_EXPERIMENTS):
        params = dict(
            phi_slope_max=random.uniform(0.002, 0.008),
            z0_max=random.uniform(150, 400),
            dR_max=random.uniform(1, 5.0),
        )
        logger.debug(params)
        graph_builder = GraphBuilder(
            indir=GRAPH_INDIR,
            outdir=Path(TemporaryDirectory().name),
            redo=True,
            measurement_mode=True,
            log_level=1,
            collect_data=False,
            **params,
        )
        graph_builder.process(stop=N_GRAPHS_MEASURED)
        measurements = graph_builder.get_measurements()
        logger.debug(measurements)
        results["results"].append(measurements | params)

    random_id = uuid.uuid1()
    MEASUREMENT_OUTDIR.mkdir(parents=True, exist_ok=True)
    with (MEASUREMENT_OUTDIR / f"results-{random_id}.json").open("w") as outf:
        json.dump(results, outf)
