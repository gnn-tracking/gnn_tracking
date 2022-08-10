from __future__ import annotations

import argparse
import itertools
import logging
import multiprocessing as mp
import os
import sys
from functools import partial

sys.path.append("../")
sys.path.append("../../")

import pickle
import time

import numpy as np
import pandas as pd
import yaml
from build_graphs import *
from utils.graph_building_utils import *
from utils.hit_processing_utils import *

# grab job ID
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(f"job {idx}")

# hyperparameters
# at 0.8 GeV: eff=98.5%, pur=34% (z0: 200, phi_slope: 0.007, dR_max: 1.7, uv_approach: 0.001)
#             eff=98.7%, pur=26% (z0: 200, phi_slope: 0.007, dR_max: 1.7, uv_approach: 0.0015)
z0_max = [275, 325, 375, 425]
phi_slope_max = [0.003, 0.004, 0.005, 0.006, 0.007]
dR_max = [1.8, 2.0, 2.2, 2.4]
uv_approach = [0.00225, 0.00275, 0.00325, 0.00375]
params = list(itertools.product(z0_max, phi_slope_max, dR_max, uv_approach))
p = params[idx]

pt_min = 0.0
pt_str = map_pt(pt_min)
input_dir = "/tigress/jdezoort/codalab/train_1"
output_dir = "optimization_results"
initialize_logger()
n_graphs = 24
file_prefixes = get_file_prefixes(
    input_dir,
    trackml=True,
    codalab=True,
    evtid_min=0,
    evtid_max=1000000,
    n_tasks=1,
    task=0,
)[:n_graphs]

with mp.Pool(processes=4) as pool:
    process_func = partial(
        process_event,
        output_dir="",
        save_graphs=False,
        pt_min=pt_min,
        n_eta_sectors=1,
        n_phi_sectors=1,
        phi_range=(-np.pi, np.pi),
        eta_range=(-4, 4),
        module_map=[],
        z0_max=p[0],
        phi_slope_max=p[1],
        dR_max=p[2],
        uv_approach_max=p[3],
        endcaps=True,
        remove_noise=True,
        remove_duplicates=False,
        use_module_map=False,
        phi_overlap=0,
        eta_overlaps=0,
        base_dir="../",
    )
    output = pool.map(process_func, file_prefixes)

stats = [out["summary_stats"] for out in output]

N = len(stats)
df = pd.DataFrame(
    {
        "ptmin": pt_min,
        "efficiency": np.mean([s["efficiency"] for s in stats]),
        "purity": np.mean([s["purity"] for s in stats]),
        "z0_max": p[0],
        "phi_slope_max": p[1],
        "dR_max": p[2],
        "uv_approach_max": p[3],
    },
    index=[idx],
)

df.to_csv(
    os.path.join(output_dir, f"ptmin-{pt_str}GeV.csv"),
    mode="a",
    index=False,
    header=False,
)
