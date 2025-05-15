INPUT_FILE="../../data/test_dataset_v3.jsonl"
RESULTS_FILE="generations_qwen25-7b-Instruct.jsonl"
OUTPUT_FILE="reasoning_results_vllm_qwen25-7b-Instruct.jsonl"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
FILE_NAME="Qwen2.5-7B-Instruct"
ANALYSIS_PATH="analysis_results/$FILE_NAME"
JUDGE_NAME="Qwen/Qwen2.5-7B-Instruct"

python llm_generation.py \
    --test_file $INPUT_FILE \
    --output_file $RESULTS_FILE \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --use_vllm \

python llm_judge.py \
    --results_file=$RESULTS_FILE \
    --output_file=$OUTPUT_FILE \
    --analysis_output_dir=$ANALYSIS_PATH \
    --use_vllm \
    --base_model=$JUDGE_NAME


# ###Finetuned###
# python llm_generation.py \
#     --test_file ../data/test_dataset_v3.jsonl \
#     --output_file generations_qwen25-7b-Instruct_lora_grpo_lr_5r-5.jsonl \
#     --base_model /n/home05/kumartanmay/main/calibrated_multilingualism/Factual_R1/evaluation_results/temp \
#     --use_vllm \


# # ##with lora lora_weights if using full finetuning
# # python llm_generation.py \
# #     --test_file ../data/test_dataset.jsonl \
# #     --output_file generations_qwen25-7b-Instruct_lora_grpo_lr_5r-5.jsonl \
# #     --base_model Qwen/Qwen2.5-7B-Instruct \
# #     --lora_weights /n/home05/kumartanmay/main/calibrated_multilingualism/Factual_R1/runs/qwen2.5-7B-R1-batch-8-lr-5e-5-template-1_lora_r_8/checkpoint-100 \
# #     --use_vllm \
# #     --merged_model_dir ./temp/

# python llm_judge.py \
#     --results_file=$RESULTS_FILE \
#     --output_file=$OUTPUT_FILE \
#     --analysis_output_dir=analysis_results \
#     --use_vllm \
#     --base_model=$MODEL_NAME


