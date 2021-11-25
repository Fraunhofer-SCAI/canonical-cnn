#!/bin/bash
#SBATCH --job-name=small_mnist
#SBATCH --output=mnist_output.txt
#SBATCH --error=mnist_error.txt
#SBATCH -p gpu

module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/LeNet/mnist.py --lr=1e-3 --epochs=1 --mode='Weight' --optimizer='SGD' --init_method='KUNIFORM'
# /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/src/ConvNet/test.py

