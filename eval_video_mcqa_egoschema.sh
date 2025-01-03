set -x

EVAL_DATA_DIR=eval_data
OUTPUT_DIR=eval_output
# CKPT_NAME="temp_videofix2"
# ADAPTER_STEPS="1000"
# CHECKPOINT_PATH="/fsx/miquel/smol-vision/smolvlm-longvumix-high_2lr_videofix/checkpoint-1000"
# MAX_FRAMES = 50
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/egoschema/answers/${CKPT_NAME}/${ADAPTER_STEPS}/merge.csv
            # --checkpoint-path ${CKPT_NAME} \

CHECKPOINT_PARAM=""
if [ "$CHECKPOINT_PATH" != "SKIP" ]; then
    CHECKPOINT_PARAM="--checkpoint-path ${CHECKPOINT_PATH}"
fi

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 eval/inference_video_mcqa_egoschema_fps.py \
            --video-folder ${EVAL_DATA_DIR}/egoschema/good_clips_git \
            --question-file ${EVAL_DATA_DIR}/egoschema/questions.json \
            --max-frames ${MAX_FRAMES} \
            --fps 2.0 \
            ${CHECKPOINT_PARAM} \
            --answer-file ${OUTPUT_DIR}/egoschema/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}.csv &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    echo 'q_uid, answer' >> "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/egoschema/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}.csv >> "$output_file"
    done
fi

python3 eval/eval_video_mcqa_egoschema_results.py eval/egoschema_subset_answers.json ${output_file}