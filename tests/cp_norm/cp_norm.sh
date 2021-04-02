#!/bin/bash
#SBATCH --job-name=Alexnet_rank_estimation
#SBATCH --output=cpnorm_anet_output_10.txt
#SBATCH --error=cpnorm_anet_error.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1


module load Anaconda3/
# Load conda environment

python tests/cp_norm/test.py 10

