#!/usr/bin/env bash

#SBATCH -o ./512_512_2_log.out # STDOUT out
#SBATCH -e ./512_512_2_log.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module add python
module load python
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

python endtoend.py --epoch=30 --batch-size=10 --learning-rate=0.0075 --momentum=0.95 --model="Basic" --outchannel-stride=32 --normalisation="minmax" --length-conv=512 --stride-conv=512