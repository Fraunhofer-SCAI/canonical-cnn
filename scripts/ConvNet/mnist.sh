#!/bin/bash
#SBATCH --job-name=small_mnist
#SBATCH --output=mnist_output.txt
#SBATCH --error=mnist_error.txt
#SBATCH -p gpu

module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/ConvNet/mnist.py --lr=1e-3 --epochs=1 --mode='CP' --optimizer='RMSPROP' --compress_rate=25 --name='./mnist_cnn_rmsprop_2.pt' --resume --save-model
# /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/src/ConvNet/test.py

