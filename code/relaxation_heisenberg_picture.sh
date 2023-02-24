#!/bin/bash
#SBATCH -J OMP-job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition extra_long
#SBATCH --time 28-0:00:00
#SBATCH --mem 2000G

# set needed CPUs per task for OMP (OMP_THREADS)
#SBATCH --cpus-per-task=8
     
# for Bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS=40
export MKL_DYNAMIC=0
cd /home/kwang/Documents/liouvillian-level-stats/code/
python relaxation_heisenberg_picture.py -l $1 -r $2

exit 0

