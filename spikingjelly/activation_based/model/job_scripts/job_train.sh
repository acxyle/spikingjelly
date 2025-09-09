#!/usr/bin/env bash
# --- add HPC resources here


head_info=" 
🔥🔥🔥🔥🔥

This script is using **sp** training script.
This script is used for (a) training on local device and (b) debug on SLURM cluster

Activate corresponding conda environment in the shell before running this script.

TODO:
- edit this script and the head_info

"

set -euo pipefail

root=/local/data/acxyle/Github/spikingjelly/spikingjelly/activation_based/model

cd $root

MODEL_ARCH=spikformer_b_16_plain
DATASET=C2k
NEURON=LIF

# --- log config (SLURM-aware)
# If running under SLURM, don't override stdout/stderr.
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_PROCID:-}" ]]; then
echo "[INFO] Detected SLURM job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}; using SLURM-managed logs."
else
short_host=$(hostname)
short_host="${short_host%%.*}"
jobid=$$

time_stamp=$(date +%Y%m%d-%H%M%S)
prefix="${short_host}_${jobid}_${time_stamp}"
output_name="sp_training_${MODEL_ARCH}_${DATASET}"
exec > >(tee -a "${root}/logs/${prefix}_${output_name}.output") 2>&1
fi

# -- put head_info into .output file
bash job_scripts/job_utils/txt_box.sh -w 80 -c '#' "$head_info"

echo "[INFO] Running for arch=$MODEL_ARCH, dataset=$DATASET, neuron=$NEURON"

# --- 
python training_script.py \
            -c spikformer_imagenet.yml \
            --arch $MODEL_ARCH \
            --dataset $DATASET \
            --neuron $NEURON \
            --data-fold-training True \
            --data-fold-index 0 \
            --clip-grad-norm 1.0

# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#     training_script.py \
#             --arch $MODEL_ARCH \
#             --dataset $DATASET \
#             --neuron $NEURON \
#             --data-fold-training True \
#             --data-fold-index 0 
