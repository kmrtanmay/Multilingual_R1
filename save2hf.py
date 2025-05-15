from datasets import Dataset
from huggingface_hub import HfApi
import json

# Set your Hugging Face username and dataset name
username = "krtanmay147"  # Replace with your HF username
dataset_name = "train-dataset-sft_v2"  # Choose a name for your dataset

# Load the shuffled JSONL file
data = []
with open("./data/train_dataset_sft_shuffled_v2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Create a Dataset object
dataset = Dataset.from_list(data)

# Push to the Hugging Face Hub
dataset.push_to_hub(
    f"{username}/{dataset_name}",
    private=True,  # Set to False if you want it to be public
    token=True  # Uses the token from huggingface-cli login
)

print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{username}/{dataset_name}")