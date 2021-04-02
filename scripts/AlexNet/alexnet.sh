#!/bin/bash
#SBATCH --job-name=svhn_alexnet
#SBATCH --output=new_alexnet_output.txt
#SBATCH --error=new_alexnet_error.txt
#SBATCH -p gpu

module load Anaconda3/5.1.0
# Load the conda environment here
echo Modules loaded....

CUDA_VISIBLE_DEVICES=7 python scripts/AlexNet/alexnet.py --lr=1e-2 --epochs=1 --optimizer='RMSPROP' --mode='CP' --compress_rate=50 --data_set='SVHN' --resume --name='path to saved'
