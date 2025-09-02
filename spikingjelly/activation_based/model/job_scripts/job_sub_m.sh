#!/bin/bash
#SBATCH --job-name=sp_mul|spikformer_b_16|C2k
#SBATCH --time=24:00:00
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%u-%j.out
##SBATCH --error=%x-%u-%j.err     # not use when debugging 


# ---
# this script is for slurm job submission on multiple nodes
# uses leeds.aire as example
# ---   

# --- load modules
module load miniforge
conda activate sp

# --- ddp config
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
PORT=12345     # random port, can be any free port

# --- job config
# args here might be override by yaml config file
# check ./models/training_configs/<arch>_<data>.yml file for more details
MODEL_ARCH=spikformer_b_16
DATASET=C2k
EPOCHS=320
BATCH_SIZE=32
LR_WARMUP_EPOCHS=10
LR=5e-5
OPT=adamw
WEIGHT_DECAY=1e-2
RDVZ_ID=$SLURM_JOB_ID
#RDVZ_ID=$RANDOM
TRAIN_CROP_SIZE=224

NEURON=LIF


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

# export LOGLEVEL=INFO     # --- for debugging
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# --- launch experiment
cd /users/sczya/Github/spikingjelly/spikingjelly/activation_based/model

for sidx in {0..0}
do
echo "==================================================="
echo "Starting fold $sidx at $(date '+%Y-%m-%d %H:%M:%S')"
fold_start_time=$(date +%s)

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$RDVZ_ID --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:$PORT training_script.py \
                                    --arch $MODEL_ARCH \
                                    --cupy \
                                    --data-fold-training True \
                                    --data-fold-index $sidx \
                                    --dataset $DATASET \
                                    -j 8 \
                                    --epochs $EPOCHS \
                                    --batch-size $BATCH_SIZE \
                                    --lr-warmup-epochs $LR_WARMUP_EPOCHS \
                                    --lr $LR \
                                    --opt $OPT \
                                    --weight-decay $WEIGHT_DECAY \
                                    --train-crop-size $TRAIN_CROP_SIZE \
                                    --neuron $NEURON \

echo "==================================================="
echo "Completed fold $sidx at $(date '+%Y-%m-%d %H:%M:%S')"
fold_end_time=$(date +%s)
fold_duration=$((fold_end_time - fold_start_time))
fold_hours=$((fold_duration / 3600))
fold_minutes=$(( (fold_duration % 3600) / 60 ))
fold_seconds=$((fold_duration % 60))
echo "Fold $sidx runtime: ${fold_hours}h ${fold_minutes}m ${fold_seconds}s"

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