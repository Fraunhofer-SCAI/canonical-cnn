#!/bin/bash
#SBATCH --job-name=svhn_alexnet
#SBATCH --output=new_alexnet_output.txt
#SBATCH --error=new_alexnet_error.txt
#SBATCH -p gpu 
#SBATCH --time=48:00:00
#SBATCH --gres gpu:v100:1

module load Anaconda3
# Load conda environment

max=11
for i in `seq 2 $max`
do
    CUDA_VISIBLE_DEVICES=3 python alexnet.py --lr=1e-3 --epochs=150 --optimizer='ADAM' --init_method='KUNIFORM' --mode='CP' --data_set='cifar10' --seed="$i"
done
