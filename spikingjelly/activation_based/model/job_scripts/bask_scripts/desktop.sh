srun --account=liuje-snn --qos=bham --time=00:30:00 --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=16 --mem-per-cpu=4G -pty /bin/bash

# check storage
mmlsquota -u yangzx --block-size=auto bask


# --- notes for GPU consumption
# --- if train large model, enable 'sync_bn' for BN calculation of bn across GPUs 
# --- for ViT, reshape the tensor if can not replace BN with LN

# spikformer_b_16: batch_size 32 -> GPU: 16.76 GB
# spikformer_b_16_plain: batch_size 32 -> GPU: 28.38 GB

# spikformer_b_32: batch_size 32 -> GPU: 8.96 GB
# spikformer_b_32_plain: batch_size 32 -> GPU: 9.69 GB

# spikformer_l_16: batch_size 32 -> GPU: 23.36 GB
# spikformer_l_16_plain: batch_size 32 -> GPU: 75.66 GB

# spikformer_l_32: batch_size 32 -> GPU: 12.84 GB
# spikformer_l_32_plain: batch_size 32 -> GPU: 24.40 GB

# spikformer_h_14: batch_size 32 -> GPU: 31.35 GB
# spikformer_h_14_plain: batch_size 6 -> GPU: --.-- GB