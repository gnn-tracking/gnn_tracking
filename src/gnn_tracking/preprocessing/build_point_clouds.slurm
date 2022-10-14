#!/bin/bash
#SBATCH --job-name=build-point-clouds  # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --cpus-per-task=1              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                      # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=2:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --array=0-500                  # total jobs in array

module purge
module load anaconda3
conda activate pyg2

echo $PWD
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Executing on machine: $(hostname)"

python build_point_clouds.py
