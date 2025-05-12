# Multilingual_R1

This project trains LLMs using DeepSeek R1 GRPO method to improve their multilingual reasoning.

---

## Dataset Preparation

First, prepare the factual question-answer dataset.

This will generate question-answer pairs across 13 languages, focusing on entities like persons, events, and landmarks linked to countries.

---

## Training Instructions

### 1. Single GPU Training

If you want to train the model using a **single GPU**, run:

```bash
bash train_single_gpu.sh
```

### 2. Distributed Training (4 A100 GPUs)

If you want to train the model using **distributed training** on 4 A100 GPUs, run:

```bash
bash main.sh
```

This script will handle launching the distributed training and saving logs automatically.

---

## Configuration Files

- `accelerate_config.yaml` : Configuration for distributed training with Accelerate.
- `factual_grpo_config.yaml` : Configuration specific to the GRPO training.

---

## Main Scripts

- `prepare_factual_dataset.py` : Script to prepare the factual recall dataset.
- `run_factual_grpo.py` : Script to launch GRPO fine-tuning.
- `launch_distributed_training.sh` : Launch script for multi-GPU training.
- `train_single_gpu.sh` : Script for single GPU training.
- `run_with_logs.sh` : Script for distributed training with logging.

---

## Outputs

- `successful_factual_responses.txt` : Log of successful factual responses after training.

---

## Notes
- Ensure the dataset is prepared **before** starting the training.
- Adjust batch size and learning rate in the YAML config files depending on GPU memory and setup.

---

