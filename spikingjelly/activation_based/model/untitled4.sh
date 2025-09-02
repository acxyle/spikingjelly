#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=liuje-snn
#SBATCH --qos=bham
#SBATCH --time 00:20:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu 4G

module purge
module load baskerville

cd /bask/projects/l/liuje-snn/acxyle
source miniconda3/bin/activate
conda activate sp

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
PORT=$((1024 + RANDOM % (65535 - 1024 + 1)))      # generage a ramdom number between 1024 and 65535 for communication

echo HEAD_NODE: $HEAD_NODE
echo SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES
echo SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE
echo SLURM_JOB_ID: $SLURM_JOB_ID

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd Github/spikingjelly/spikingjelly/activation_based/model

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:$PORT train_imagenet_example.py --output-dir ./logs_resnet101 --model resnet101
