#!/bin/bash

# Setup environment
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

# Config file
CONFIG_FILE="factual_grpo_config.yaml"
OUTPUT_DIR="runs/qwen2.5-3B-R1-factual-qa_template_1_lora"

# Distributed training settings
NUM_GPUS=2
MASTER_PORT=$(shuf -i 10000-65535 -n 1)

# Memory optimizations for distributed training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# Configure NCCL for standard multi-GPU setup
export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=ipv4  # Force IPv4 instead of IPv6

# Display configuration
echo "Launching distributed training on $NUM_GPUS GPUs"
echo "Using configuration from $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Master port: $MASTER_PORT"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Launch distributed training using accelerate
exec python -m accelerate.commands.launch \
    --config_file accelerate_config.yaml \
    --num_processes=$NUM_GPUS \
    --main_process_port=$MASTER_PORT \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    run_factual_grpo.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR