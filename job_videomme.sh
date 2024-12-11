#!/bin/bash
#SBATCH --job-name=smolvlm_eval-videomme  # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (typically one per node)
#SBATCH --gres=gpu:1                   # Number of GPUs per node
##SBATCH --mem=128G                      # Memory per node
#SBATCH --output=/fsx/miquel/smol-vision/eval_logs/evals-videomme-%x-%j.out            # Standard output log
#SBATCH --error=/fsx/miquel/smol-vision/eval_logs/evals-videomme-%x-%j.err             # Standard error log
#SBATCH --partition=hopper-prod
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miquel.farre@huggingface.co



set -x -e
source ~/.bashrc
export HF_HOME=/fsx/miquel/cache
cd /fsx/miquel/smol-vision
source /fsx/miquel/miniconda3/etc/profile.d/conda.sh
conda activate smolvlm
export LD_LIBRARY_PATH=/fsx/miquel/miniconda3/envs/evaluator/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# export CKPT_NAME="temp_videofix2"
# export ADAPTER_STEPS="1000"
# export CHECKPOINT_PATH="/fsx/miquel/smol-vision/smolvlm-longvumix-high_2lr_videofix/checkpoint-1000"
# export MAX_FRAMES=50

# Run eval scripts
# Run eval scripts
srun bash eval_video_mcqa_videomme.sh 


