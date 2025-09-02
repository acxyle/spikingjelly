#!/bin/bash
## Created on Fri Jun 13 18:15:36 2025
##@author: sczya
## ---
## this code does not contain old distributed.launch
## e.g.
##  --- old 
#        python -m torch.distributed.launch --nproc_per_node=3 train.py
#   --- new 
#       torchrun --nproc_per_node=3 train.py

## ---- method 1 -----
## Directly launching torchrun
##

## ssh into different computation nodes and launch below command with different <node_rank>
## The pytorch tutorial[1] advises using one node with broad bandwidth for between-nodes communication 
## a lot of practices choose to use the rank 0 node as the communication node
## The nodes launched first will hang until the final node has been allocated a job

# ---
srun --account liuje-snn --qos bham --nodes 2 --tasks-per-node 1 --gpus-per-task 1 --cpus-per-task 8 --mem-per-cpu 4G --time 00:20:00 --pty /bin/bash
# ---

torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=0 \
--rdvz_id=RANDOM \
--rdvz_backend=c10d \
--rdvz_endpoint=hostname:ip \
training_script.py 


## ----- method 2 -----
## Slurm job script
## 

## modify the below job script for different Slrum version and system requirement

#! /bin/bash
#SBATCH --job-name=multi_nodes_training
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

nodes=($(scontrol show hostname $SLURM_JOB_NODELIST))
head_node=${nodes[0]}
#head_node_ip=$(hostname --ip --ip-address)
#echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 2 \ 
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_endpoint $head_node:10233 \
training_script.py


## --- references
## 1. 
## https://www.youtube.com/watch?v=KaAJtI1T2x4&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj&index=5
## 2. 
## https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
## 3.
## https://github.com/stanford-cs336