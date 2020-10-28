#!/bin/bash
#
#SBATCH --job-name=reconstruction
#SBATCH --output=reconst_output.txt
#SBATCH --error=reconst_error.txt
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16

module load Anaconda3/5.1.0
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env
echo Modules loaded....

CUDA_VISIBLE_DEVICES=4,5 /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/workbench/CP_decomposition/Python_files/reconstruction_error.py

/reconstruction_error.py