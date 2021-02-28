#!/bin/bash
#SBATCH --job-name=Alexnet_rank_estimation
#SBATCH --output=cpnorm_anet_output_10.txt
#SBATCH --error=cpnorm_anet_error.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1


module load Anaconda3/
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/src/cp_norm/test.py 10

