#!/bin/bash
#SBATCH -J OMP-job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1


# set needed CPUs per task for OMP (OMP_THREADS)
#SBATCH --cpus-per-task=8
     
# for Bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd /home/kwang/Documents/Random\ Liouvillians/data/job_runs 
python ../../code/cluster.py

exit 0

