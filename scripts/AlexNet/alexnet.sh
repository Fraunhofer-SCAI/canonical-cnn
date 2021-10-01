#!/bin/bash
#SBATCH --job-name=svhn_alexnet
#SBATCH --output=new_alexnet_output.txt
#SBATCH --error=new_alexnet_error.txt
#SBATCH -p gpu

module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env


CUDA_VISIBLE_DEVICES=7 /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/AlexNet/alexnet.py --lr=0.01 --epochs=20 --optimizer='SGD' --mode='Tai' --resume --name='./test_runs/exp1_oldnw/model_best.pth.tar' --configpath='./config.json'

