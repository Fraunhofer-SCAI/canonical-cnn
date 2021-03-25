#!/bin/bash
#SBATCH --job-name=svhn_alexnet
#SBATCH --output=new_alexnet_output.txt
#SBATCH --error=new_alexnet_error.txt
#SBATCH -p gpu

module load Anaconda3/5.1.0
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env
echo Modules loaded....

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/AlexNet/alexnet_transfered.py --lr=1e-2 --epochs=1 --optimizer=0 --mode=1 --compress_rate=0 --data_set='SVHN'
