#!/bin/bash
#SBATCH -J OMP-job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition medium
#SBATCH --time 2-0:00:00


# set needed CPUs per task for OMP (OMP_THREADS)
#SBATCH --cpus-per-task=8
     
# for Bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd /home/kwang/Documents/Random\ Liouvillians/data/match_runs/
python /home/kwang/Documents/Random\ Liouvillians/code/match_denisov.py

exit 0

