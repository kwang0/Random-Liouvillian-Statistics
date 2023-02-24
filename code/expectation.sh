#!/bin/bash
#SBATCH -J OMP-job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem 500G
#SBATCH --partition medium
#SBATCH --time 2-0:00:00

# set needed CPUs per task for OMP (OMP_THREADS)
#SBATCH --cpus-per-task=40

# for Bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS=40
export MKL_DYNAMIC=0

python /home/kwang/Documents/Random\ Liouvillians/code/expectation.py

exit 0
