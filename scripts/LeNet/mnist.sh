#!/bin/bash
#SBATCH --job-name=small_mnist
#SBATCH --output=mnist_output.txt
#SBATCH --error=mnist_error.txt
#SBATCH -p gpu

module load Anaconda3
# Load the conda environment herev

python scripts/LeNet/mnist.py --lr=1e-3 --epochs=1 --mode='CP' --optimizer='RMSPROP' --compress_rate=25 --name='path to saved model' --resume --save-model

