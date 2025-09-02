#!/bin/bash 
#SBATCH --job-name=sp_sin|spikformer_b_16-LIF|C2k
#SBATCH --time=24:00:00     # Request runtime (hh:mm:ss)             
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%u-%j.out
##SBATCH --gres=gpu:3   


# ---
# this script is for slurm job submission on single node
# uses leeds.aire as example
# ---   

# ---
MODEL_ARCH=spikformer_b_16
DATASET=C2k
NEURON=LIF

# Load any necessary modules
module load miniforge
module load apptainer
conda activate sp

# Run the job
cd /users/sczya/Github/spikingjelly/spikingjelly/activation_based/model

torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    training_script.py \
            --arch $MODEL_ARCH \
            --dataset $DATASET \
            --neuron $NEURON \
            --data-fold-training True \
            --data-fold-index 0 

# --- hyperparameters for different datasets and models need to be set other than defaults
# spiking_vgg16_bn:
#   IF: --wd 1e-4
