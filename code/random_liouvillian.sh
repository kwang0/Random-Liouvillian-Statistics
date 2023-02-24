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
echo $SLURM_CPUS_PER_TASK > num_cpus.txt
cd /data3/kwang
python /home/kwang/Documents/Random\ Liouvillians//code/random_liouvillian.py -l $1 -o $2 -r $3 > L${1}_op${2}_run${3}.txt

exit 0

