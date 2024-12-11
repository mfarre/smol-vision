set -x

EVAL_DATA_DIR=eval_data
OUTPUT_DIR=eval_output
CKPT_NAME="Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH="/fsx/miquel/simplevideo_trl/simplevideo_trl/qloraqkvoprojllamafactoryShorts/final_model"
ADAPTER_STEPS="final"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/perception_test_mcqa/answers/${CKPT_NAME}/${ADAPTER_STEPS}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")    
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 eval/inference_video_mcqa_perception_test_mcqa.py \
            --model-path ${CKPT_NAME} \
            --video-folder ${EVAL_DATA_DIR}/perception_test_mcqa/videos \
            --question-file ${EVAL_DATA_DIR}/perception_test_mcqa/mc_question_test.json \
            --answer-file ${OUTPUT_DIR}/perception_test_mcqa/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --adapter-path ${ADAPTER_PATH} \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    echo "{" >> "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/perception_test_mcqa/answers/${CKPT_NAME}/${ADAPTER_STEPS}/${CHUNKS}_${IDX}.json >> "$output_file"
    done

    sed -i '$s/.$//' $output_file

    echo "}" >> "$output_file"
fi