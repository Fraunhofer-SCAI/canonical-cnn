#!/bin/bash
#
#SBATCH --job-name=alexnet
#SBATCH --output=new_alexnet_output.txt
#SBATCH --error=new_alexnet_error.txt
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16

module load Anaconda3/5.1.0
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env
echo Modules loaded....

CUDA_VISIBLE_DEVICES=4,5 /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/main/small_alexnet/alexnet_transfered.py --lr=1e-3 --momentum=0.1 --epochs=320 --rank=140
