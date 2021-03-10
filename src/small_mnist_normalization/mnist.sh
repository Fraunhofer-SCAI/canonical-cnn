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
#module load Singularity
#singularity exec --nv /opt/software/Singularity/pytorch-19.09-py3.sif python metafile.py
#python metafile.py
#source /home/lveeramacheneni/.bashrc
#conda activate /home/lveeramacheneni/lconda_env
#echo Modules loaded....
/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/src/small_mnist/mnist.py --lr=1e-3 --epochs=50 --mode=1 --optimizer=1
#/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/src/small_mnist/metafile.py

