#!/bin/bash
#SBATCH --job-name=small_mnist
#SBATCH --output=mnist_output_1.txt
#SBATCH --error=mnist_error_1.txt
#SBATCH -p gpu

module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env

CUDA_VISIBLE_DEVICES=7 /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/ImageNet/main.py --lr=0.1 --epochs=100 --dataset='tinyimagenet'
