#!/bin/bash
#SBATCH --job-name=small_mnist
#SBATCH --output=mnist_output.txt
#SBATCH --error=mnist_error.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1



module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/src/ConvNet/mnist.py --lr=1e-3 --epochs=20 --mode=1 --optimizer=0 --compress_rate=75

