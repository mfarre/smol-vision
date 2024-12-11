#!/bin/bash
#SBATCH --job-name=quen2_eval-perceptiontest  # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (typically one per node)
#SBATCH --gres=gpu:1                   # Number of GPUs per node
##SBATCH --mem=128G                      # Memory per node
#SBATCH --output=logs/evals-perceptiontest-%x-%j.out            # Standard output log
#SBATCH --error=logs/evals-perceptiontest-%x-%j.err             # Standard error log
#SBATCH --partition=hopper-prod
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miquel.farre@huggingface.co


# Set up the build environment
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Ensure PyTorch library path is included
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__path__[0]), 'lib'))"):$LD_LIBRARY_PATH


# Load any required modules
module load cuda/12.1

set -x -e
source ~/.bashrc
export HF_HOME=/fsx/miquel/cache
cd /fsx/miquel/qwenexperiments
source /fsx/miquel/miniconda3/etc/profile.d/conda.sh
conda activate qwen

# Run eval scripts
srun bash eval_video_mcqa_perception_test_mcqa.sh


