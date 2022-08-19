#!/bin/bash
#SBATCH --job-name=pc-job     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-757              # job array with index values 0, 1, 2, 3, 4

echo $PWD
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Executing on machine: $(hostname)"

module purge
module load anaconda3
conda activate pyg2

python scan_point_cloud_construction.py
