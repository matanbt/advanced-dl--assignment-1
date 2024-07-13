#! /bin/sh

#SBATCH --job-name=exp-lo
#SBATCH --output=logs/exp-lo.out # redirect stdout
#SBATCH --error=logs/exp-lo.err # redirect stderr
#SBATCH --partition=gpu-a100-killable
#SBATCH --time=1300 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=20G # CPU memory (MB)
#SBATCH --gpus=1

jupyter nbconvert --to html --execute --output=exp-lo.html ExperimentingListOps.ipynb
