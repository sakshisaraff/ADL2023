#!/usr/bin/env bash

#SBATCH --job-name=coursework
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --account=COMS030144
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=16GB

module purge
module add python
module load python
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

##uncomment to run the basic CNN model with length and stride of 256 for strided convolution
#python endtoend.py --model="Basic"

##uncomment to run the basic CNN model with different lengths and strides for the strided convolution
##change the values input into --length-conv and --stride-conv according to the lengths and strides, respectively
#python endtoend.py --model="Basic" --length-conv=1024 --stride-conv=512

##uncomment to run the basic extension to the CNN model- group norm and dropout
#python endtoend.py --epoch=40 --model="Extension1"

##uncomment to run the deep CNN model- extension part 2
##takes 5 hours to run so please have --time around 6 or above
#python endtoend.py --model="Deep"