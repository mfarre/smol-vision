#!/bin/bash
#SBATCH --job-name=smollvlmftvideo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # Keep <=64 to enable "mix" sharing of resources
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --output=/fsx/miquel/smol-vision/slurmlogs/%x-%j.out
#SBATCH --err=/fsx/miquel/smol-vision/slurmlogs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miquel.farre@huggingface.co

set -x -e
# source ~/.bashrc
export HF_HOME=/fsx/miquel/cache

source /fsx/miquel/miniconda3/etc/profile.d/conda.sh 
conda activate smolvlm
which torchrun


echo "Python location: $(which python)"
export LD_LIBRARY_PATH=/fsx/miquel/miniconda3/envs/evaluator/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python -c "import sys; print('Python path:', sys.path)"
python -c "import torch; print('Torch location:', torch.__file__)"
python -c "import transformers; print('Transformers location:', transformers.__file__)"



# Environment setup
export RANK=$SLURM_PROCID
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6001

# # Create hostfile for DeepSpeed
# HOSTFILE=/tmp/hostfile_$$
# scontrol show hostnames $SLURM_JOB_NODELIST | while read -r host; do
#     echo "$host slots=8"
# done > $HOSTFILE

# echo "Generated hostfile:"
# cat $HOSTFILE

# Set deepspeed environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO

# # Launch with DeepSpeed
# deepspeed \
#     --hostfile $HOSTFILE \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --num_gpus $SLURM_NTASKS_PER_NODE \
#     --num_nodes $SLURM_NNODES \
#     Idefics3_FT.py \
#     --max_frames 50 \
#     --wandb \
#     --deepspeed ds_config.json

# # Cleanup
# rm $HOSTFILE

# Print debug info
echo "== Distributed setup =="
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "===================="

# Launch training
srun --label \
    python -u Idefics3_FT_ds.py \
        --max_frames 50 \
        --wandb \
        --deepspeed ds_config.json \
        --local_rank $LOCAL_RANK