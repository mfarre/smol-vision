#!/bin/bash
#SBATCH --job-name=smollvlmftvideo
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --output=/fsx/miquel/smol-vision/slurmlogs/%x-%j.out
#SBATCH --error=/fsx/miquel/smol-vision/slurmlogs/%x-%j.err

set -xe
echo "START TIME: $(date)"

# Set MASTER_ADDR and MASTER_PORT for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_PORT=6001

if [ -z "$WANDB_RUN_ID" ]; then
    export WANDB_RUN_ID=$(tr -dc 'a-z0-9' </dev/urandom | head -c 8)
fi
echo "WANDB_RUN_ID: $WANDB_RUN_ID"

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))


export HF_HOME=/fsx/miquel/cache
source /fsx/miquel/miniconda3/etc/profile.d/conda.sh
conda activate smolvlm

srun accelerate launch \
     --machine_rank=$SLURM_NODEID \
     --main_process_ip=$MASTER_ADDR \
     --main_process_port=$MASTER_PORT \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
     Idefics3_FT_ds.py --wandb

echo "END TIME: $(date)"