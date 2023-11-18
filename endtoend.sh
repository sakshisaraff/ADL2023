#!/usr/bin/env bash

#SBATCH --job-name=coursework
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH --account=COMS030144
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --mem=4GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module add python
module load python
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

python end_to_end.py