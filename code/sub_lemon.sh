#!/bin/bash
#SBATCH -J OMP-job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition long
#SBATCH --time 14-0:00:00
#SBATCH --mem 500G

# set needed CPUs per task for OMP (OMP_THREADS)
#SBATCH --cpus-per-task=40
     
# for Bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS=40
export MKL_DYNAMIC=0
cd /home/kwang/Documents/liouvillian-level-stats/code/
python /home/kwang/Documents/liouvillian-level-stats/code/sub_lemon.py -l $1 -o $2 -r $3
exit 0

