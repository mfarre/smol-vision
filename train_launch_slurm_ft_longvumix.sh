#!/bin/bash
#SBATCH --job-name=smollvlmftvideo
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # Keep <=64 to enable "mix" sharing of resources
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
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



# srun python -m torch.distributed.launch  --nproc_per_node=8 --use_env SmolVLM_FT_video.py 
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6001          # Set an open port for communication
export WORLD_SIZE=$(($SLURM_NNODES * 8))   # Total number of GPUs across nodes
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=0

# Print Python path for debugging
python -c "import sys; print('Python path:', sys.path)"
python -c "import torch; print('PyTorch path:', torch.__file__)"


srun torchrun \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    SmolVLM_FT_2.py --max_frames 50 --temporal_tokens --wandb