import os
import sys

sys.path.append("../")

import itertools

import run_track_condensation_network

# grab job ID
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

# hyperparameters
q_min = [0.01, 0.1, 1]
sb = [0.01, 0.1, 1]
lr = [0.0001, 0.001, 0.01]
params = list(itertools.product(q_min, sb, lr))
params = params[idx]
print(f"(q_min, sb, lr)={params}")

args = [
    "--indir",
    "../graphs/train1_ptmin1",
    "--outdir",
    f"../trained_models",
    "--stat-outfile",
    f"train_stats/train1_ptmin1_{idx}.csv",
    "--n-train",
    "8000",
    "--n-test",
    "2000",
    "--n-val",
    "250",
    "--q-min",
    str(params[0]),
    "--sb",
    str(params[1]),
    "--learning-rate",
    str(params[2]),
]
run_track_condensation_network.main(args)
