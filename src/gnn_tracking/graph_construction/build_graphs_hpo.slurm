#!/bin/bash

#SBATCH --job-name=build-graphs-hpo    # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --cpus-per-task=1              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                      # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=2:32:00                 # total run time limit (HH:MM:SS)
#SBATCH --output=build-graphs-hpo-%j.log

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

python build_graphs_hpo.py

echo "Finished"
