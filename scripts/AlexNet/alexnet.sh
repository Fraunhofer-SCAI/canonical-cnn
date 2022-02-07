#!/bin/bash
#SBATCH --job-name=svhn_alexnet
#SBATCH --output=new_alexnet_output.txt
#SBATCH --error=new_alexnet_error.txt
#SBATCH -p gpu 
#SBATCH --time=48:00:00
#SBATCH --gres gpu:v100:1

module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env
#
#
/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/AlexNet/alexnet.py --lr=1e-3 --epochs=150 --optimizer='ADAM' --init_method='CPD' --mode='CP' --data_set='cifar10' --seed=1


max=10
for i in `seq 2 $max`
do
##    /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/AlexNet/alexnet.py --lr=1e-4 --epochs=150 --optimizer='RMSPROP' --init_method='KNORMAL' --mode='None' --data_set='cifar100' --seed="$i"
    CUDA_VISIBLE_DEVICES=3 /home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/network-compression/scripts/AlexNet/alexnet.py --lr=1e-3 --epochs=150 --optimizer='ADAM' --init_method='KUNIFORM' --mode='CP' --data_set='cifar10' --seed="$i"
done
