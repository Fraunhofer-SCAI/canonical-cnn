#!/bin/bash
#
#SBATCH --job-name=alexnet
#SBATCH --output=alexnet_output.txt
#SBATCH --error=alexnet_error.txt
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16

module load Anaconda3/5.1.0
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env
echo Modules loaded....

CUDA_VISIBLE_DEVICES=4,5 /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/main/alexnet.py
/alexnet.py