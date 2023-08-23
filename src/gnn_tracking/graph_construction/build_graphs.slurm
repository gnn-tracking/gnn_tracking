#!/bin/bash

# To submit jobs for every stream:
# for i in {1..9}; do export PART=$i; sbatch build_graphs.slurm; done

#SBATCH --job-name=build-graphs        # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --cpus-per-task=1              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=45G                      # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=3:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --output=build-graphs-%A-%a.log
#SBATCH --array=0-20 # total jobs in array

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

#module purge
#module load anaconda3
#conda activate pyg2

# Allow to set the stream number via environment variable
# to loop over the batch script submission
PART=${PART:-1}

echo $PWD
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Executing on machine: $(hostname)"
echo "Part: ${PART}"

python build_graphs.py \
  --indir /scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v2/part_${PART} \
  --outdir /scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v3/part_${PART} \
  --batch-size 48

echo "Finished"
