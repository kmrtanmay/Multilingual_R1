#!/bin/bash

# Extremely conservative settings for testing with a single GPU/MIG instance

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false

# Choose just one GPU - select the MIG instance from GPU 0
export CUDA_VISIBLE_DEVICES=0

# Disable distributed and use direct device assignment
export USE_TORCH_DDP=0

# Output directory
OUTPUT_DIR="runs/single-gpu-safe-test_v2"
mkdir -p $OUTPUT_DIR

echo "Starting single GPU training with safe settings..."
echo "Output directory: $OUTPUT_DIR"

# Create log directory
mkdir -p logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/safe_single_gpu_${TIMESTAMP}.log"

# Override critical settings on command line to ensure they're applied
python run_factual_grpo.py \
    --config factual_grpo_config.yaml \
    --output_dir $OUTPUT_DIR \
    --tf32 false \
    --use_vllm false \
    --num_generations 2 \
    --max_completion_length 256 \
    --gradient_accumulation_steps 16 \
    2>&1 | tee ${LOG_FILE}

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code ${PIPESTATUS[0]}"
    echo "Check ${LOG_FILE} for details"
fi