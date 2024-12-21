set -x

EVAL_DATA_DIR=eval_data
OUTPUT_DIR=eval_output
# CKPT_NAME="temp_videofix2"
# ADAPTER_STEPS="1000"
# CHECKPOINT_PATH="/fsx/miquel/smol-vision/smolvlm-longvumix-high_2lr_videofix/checkpoint-1000"
# MAX_FRAMES = 50

#CKPT_NAME="V1-filter-lowlrhighwarmup_2"
#ADAPTER_STEPS="4000"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${ADAPTER_STEPS}/merge.json
output_sub_file=${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${ADAPTER_STEPS}/merge_sub.json

CHECKPOINT_PARAM=""
if [ "$CHECKPOINT_PATH" != "SKIP" ]; then
    CHECKPOINT_PARAM="--checkpoint-path ${CHECKPOINT_PATH}"
fi

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${ADAPTER_STEPS}/*.json
fi


if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        #--checkpoint-path "/fsx/miquel/smol-vision/smolvlm-longvumix-filter1-lowlrhighwarm_2/checkpoint-4000" \
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 eval/inference_video_mcqa_videomme.py \
            --video-folder ${EVAL_DATA_DIR}/videomme/videos \
            --subtitle-folder ${EVAL_DATA_DIR}/videomme/subtitles \
            --question-file ${EVAL_DATA_DIR}/videomme/test-00000-of-00001.parquet \
            --max-frames ${MAX_FRAMES} \
            ${CHECKPOINT_PARAM} \
            --answer-file ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}.json &
    done

    wait

    # # Clear out the output file if it exists.
    # > "$output_file"

    # echo "[" >> "$output_file"

    # #Loop through the indices and concatenate each file.
    # for IDX in $(seq 0 $((CHUNKS-1))); do
    #     cat ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}.json >> "$output_file"
    # done

    # sed -i '$s/.$//' $output_file

    # echo "]" >> "$output_file"

    # # Clear out the output file if it exists.
    # > "$output_sub_file"

    # echo "[" >> "$output_sub_file"

    # #Loop through the indices and concatenate each file.
    # for IDX in $(seq 0 $((CHUNKS-1))); do
    #     cat ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}_sub.json >> "$output_sub_file"
    # done

    # sed -i '$s/.$//' $output_sub_file

    # echo "]" >> "$output_sub_file"
fi


python eval/eval_video_mcqa_videomme.py \
    --results_file ${OUTPUT_DIR}/videomme/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}.json \
    --video_duration_type "short,medium,long" \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy \
    --skip_missing \
echo "Checkpoint Path: $CHECKPOINT_PATH"

# python eval/eval_video_mcqa_videomme.py \
#     --results_file $output_sub_file \
#     --video_duration_type "short,medium,long" \
#     --return_categories_accuracy \
#     --return_sub_categories_accuracy \
#     --return_task_types_accuracy \
#     --skip_missing \
