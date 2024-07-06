#! /bin/sh

#SBATCH --job-name=exp-lo
#SBATCH --output=logs/exp-lo.out # redirect stdout
#SBATCH --error=logs/exp-lo.err # redirect stderr
#SBATCH --partition=killable  #killable  #studentbatch # (see resources section)
#SBATCH --time=1300 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=20G # CPU memory (MB)
#SBATCH --gpus=1

export PYTHONPATH="$PWD"  # set the Python's path to the current path
export HF_HOME="/home/sharifm/students/matanbentov"  # modify to home defaco dir

jupyter nbconvert --to html --execute --output=exp-lo.html ExperimentingListOps.ipynb
