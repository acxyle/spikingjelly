#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=liuje-snn
#SBATCH --qos=bham
#SBATCH --time 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
##SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu 4G

module purge
module load baskerville

cd /bask/projects/l/liuje-snn/acxyle
source miniconda3/bin/activate
conda activate sp

cd Github/spikingjelly/spikingjelly/activation_based/model

torchrun --standalone --nnodes=1 --nproc_per_node gpu train_imagenet_example.py --model resnet101