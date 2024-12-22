#!/bin/bash
# launcher.sh

# BASE_NAME="frames8_no_temp_lr_1e-5"
# BASE_NAME="frames8_with_temp_lr_1e-5"
# BASE_NAME="frames50_with_temp_lr_1e-5"
MAX_FRAMES=50
# BASE_NAME="frames${MAX_FRAMES}_no_temp_lr_1e-5"
# BASE_NAME="frames${MAX_FRAMES}_with_temp_lr_1e-5"
BASE_NAME="frames${MAX_FRAMES}_no_temp_lr_1e-5"

CHECKPOINT_BASE="/fsx/miquel/smol-vision/smolvlm_longvucauldron_${BASE_NAME}/checkpoint-"
STEPS=(250 500 1000 1250 1500 1736)

for step in "${STEPS[@]}"; do
    echo "Submitting egoschema job ${CHECKPOINT_BASE}${step}"
    
    sbatch \
        --export=ALL,CKPT_NAME="${BASE_NAME}",ADAPTER_STEPS="${step}",CHECKPOINT_PATH="${CHECKPOINT_BASE}${step}",MAX_FRAMES=${MAX_FRAMES} \
        job_egoschema.sh
    
    sleep 2
done


for step in "${STEPS[@]}"; do
    echo "Submitting mvbench job ${CHECKPOINT_BASE}${step}"
    
    sbatch \
        --export=ALL,CKPT_NAME="${BASE_NAME}",ADAPTER_STEPS="${step}",CHECKPOINT_PATH="${CHECKPOINT_BASE}${step}",MAX_FRAMES=${MAX_FRAMES} \
        job_mvbench.sh
    
    sleep 2
done


for step in "${STEPS[@]}"; do
    echo "Submitting videomme job ${CHECKPOINT_BASE}${step}"
    
    sbatch \
        --export=ALL,CKPT_NAME="${BASE_NAME}",ADAPTER_STEPS="${step}",CHECKPOINT_PATH="${CHECKPOINT_BASE}${step}",MAX_FRAMES=${MAX_FRAMES} \
        job_videomme.sh
    
    sleep 2
done


# Launch vanilla
    # sbatch \
    #     --export=ALL,CKPT_NAME="vanilla-new",ADAPTER_STEPS="vanilla-new",CHECKPOINT_PATH="SKIP",MAX_FRAMES=8 \
    #     job_videomme.sh
    # sbatch \
    #     --export=ALL,CKPT_NAME="vanilla-new",ADAPTER_STEPS="vanilla-new",CHECKPOINT_PATH="SKIP",MAX_FRAMES=50 \
    #     job_videomme.sh

    # sbatch \
    #     --export=ALL,CKPT_NAME="vanilla-new",ADAPTER_STEPS="vanilla-new",CHECKPOINT_PATH="SKIP",MAX_FRAMES=8 \
    #     job_mvbench.sh
    # sbatch \
    #     --export=ALL,CKPT_NAME="vanilla-new",ADAPTER_STEPS="vanilla-new",CHECKPOINT_PATH="SKIP",MAX_FRAMES=50 \
    #     job_mvbench.sh

    # sbatch \
    #     --export=ALL,CKPT_NAME="vanilla-new",ADAPTER_STEPS="vanilla-new",CHECKPOINT_PATH="SKIP",MAX_FRAMES=8 \
    #     job_egoschema.sh
    # sbatch \
    #     --export=ALL,CKPT_NAME="vanilla-new",ADAPTER_STEPS="vanilla-new",CHECKPOINT_PATH="SKIP",MAX_FRAMES=50 \
    #     job_egoschema.sh