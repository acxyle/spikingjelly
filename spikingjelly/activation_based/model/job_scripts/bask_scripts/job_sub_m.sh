#!/bin/bash
#SBATCH --job-name=sp_mul|spikformer_b_16_plain-LIF|vgg_face2
#SBATCH --account=liuje-snn
#SBATCH --qos=bham
#SBATCH --time=90:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu 4G
#SBATCH --partition=baskerville-a100_40
#SBATCH --output=%x-%u-%j.out
##SBATCH --error=%x-%u-%j.err

module purge
module load baskerville

cd /bask/projects/l/liuje-snn/acxyle
source miniconda3/bin/activate
conda activate sp

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
PORT=$((1024 + RANDOM % (29500 - 1024 + 1)))      # generage a ramdom number between 1024 and 29500 for communication
RDZV_ID=$SLURM_JOB_ID
#RDZV_ID=$RANDOM

# experiment config
NEURON=LIF

MODEL_ARCH=spikformer_b_16_plain
DATADIR=/bask/projects/l/liuje-snn/acxyle/Datasets
DATASET=vgg_face2/train
EPOCHS=300
BATCH_SIZE=32
LR_WARMUP_EPOCHS=20
LR=5e-5
OPT=adamw
WEIGHT_DECAY=5e-2
# TRAIN_CROP_SIZE=224


echo "========== JOB START =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
echo "Requested wall‑time: $TIME_LIMIT"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "Model_arch: $MODEL_ARCH"
echo "Dataset: $DATASET"
echo "======================================"
START_TIME=$(date +%s)

echo HEAD_NODE: $HEAD_NODE
echo SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES
echo SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE
echo SLURM_JOB_ID: $SLURM_JOB_ID

#export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

cd Github/spikingjelly/spikingjelly/activation_based/model

for ski in {0..0}
do
echo "==================================================="
echo "Starting fold $ski at $(date '+%Y-%m-%d %H:%M:%S')"
fold_start_time=$(date +%s)

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES \
				--nproc_per_node=$SLURM_GPUS_ON_NODE \
				--rdzv_id=$RDZV_ID \
				--rdzv_backend=c10d \
				--rdzv_endpoint=$HEAD_NODE:$PORT \
					training_script.py \
						-c spikformer_imagenet.yml \
						--arch $MODEL_ARCH \
						--datadir $DATADIR \
						--dataset $DATASET \
						--lr $LR \
						--weight-decay $WEIGHT_DECAY \
						--opt $OPT \
						--epochs $EPOCHS \
						--batch-size $BATCH_SIZE \
						--lr-warmup-epochs $LR_WARMUP_EPOCHS \
						--neuron $NEURON \
						--cupy \
						--data-fold-training True \
						--data-fold-index $ski \
						--clip-grad-norm 1.0 \
						
						
# ===== variable cache
# --sync-bn \
# --resume latest
# --clip-grad-norm 1.0 \

echo "==================================================="
echo "Completed fold $ski at $(date '+%Y-%m-%d %H:%M:%S')"
fold_end_time=$(date +%s)
fold_duration=$((fold_end_time - fold_start_time))
fold_hours=$((fold_duration / 3600))
fold_minutes=$(( (fold_duration % 3600) / 60 ))
fold_seconds=$((fold_duration % 60))
echo "Fold $ski runtime: ${fold_hours}h ${fold_minutes}m ${fold_seconds}s"

done

# === Job completion information
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "========== JOB COMPLETED =========="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "===================================="
