#!/bin/bash
#SBATCH --job-name=sp_sin|spikformer|C2k
#SBATCH --account=liuje-snn
#SBATCH --qos=bham
#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=3
#SBATCH --partition=baskerville-a100_40
#SBATCH --cpus-per-task=16 
#SBATCH --mem-per-cpu 4G
#SBATCH --output=%x-%u-%j.out
##SBATCH --error=%x-%u-%j.err
##SBATCH --constraint=a100_40

# experiment config
MODEL_ARCH=spikformer
DATASET=C2k

# --- empirical time consumption for training task on Baskerville:
# --- every 300 epochs, <days-hours:minutes>
# Data(samples)     |     Arch     |     1 GPU     |     1 Node (4 GPUs)
# C2k (80k)         |   vgg16_bn   |     12:00     |     3:00
# 
# --- estimate the required time by above table
# --- suggest to use job_sub_m.sh for multiple node training


echo HEAD_NODE: $HEAD_NODE
echo SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES
echo SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE
echo SLURM_JOB_ID: $SLURM_JOB_ID

module purge
module load baskerville

cd /bask/projects/l/liuje-snn/acxyle
source miniconda3/bin/activate
conda activate sp

cd Github/spikingjelly/spikingjelly/activation_based/model

echo "===== job started ====="

for ski in {0..0}
do
torchrun --standalone --nnodes=1 --nproc-per-node=gpu training_script.py --arch $MODEL_ARCH --data-fold-training True --data-fold-index=$ski --lr 5e-4 --dataset $DATASET --weight-decay 5e-2 --opt adamw --resume latest
done

echo "===== all jobs completed ====="
