#!/bin/bash
#
#SBATCH --job-name=alexnet
#SBATCH --output=new_alexnet_output.txt
#SBATCH --error=new_alexnet_error.txt
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH -n 1

module load Anaconda3/5.1.0
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env
echo Modules loaded....

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/src/alexnet/alexnet_transfered.py --lr=1e-3 --momentum=0.0 --epochs=150 --optimizer=1 --mode=1
