#!/bin/bash
#SBATCH --job-name=small_mnist
#SBATCH --output=mnist_output.txt
#SBATCH --error=mnist_error.txt
#SBATCH -p gpu

module load Anaconda3
# load conda encironemtn

python mnist.py --lr=1e-3 --epochs=1 --mode='Weight' --optimizer='SGD' --init_method='KUNIFORM'

