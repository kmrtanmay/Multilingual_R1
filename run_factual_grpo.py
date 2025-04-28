import logging
import os
from dataclasses import dataclass
from datetime import datetime
import re
import random
import torch
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = None
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Reward functions
########################

def format_reward_func_factual(completions, target, **kwargs):
    """
    Checks whether completion follows the format: <think>...</think>\n<answer>...</answer>
    
    Args:
        completions (list[str])
        target (list[str]) - unused but kept for API compatibility

    Returns:
        list[float]
    """
    rewards = []
    for completion in completions:
        try:
            completion = "<think>" + completion  # pre-add <think> if you assume assistant always starts inside thinking
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
                
            # Log samples occasionally
            if random.random() < 0.05:  # 5% chance to log samples
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "factual_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        except Exception:
            rewards.append(0.0)
    return rewards


def correctness_reward_func_factual(completions, targets=None, **kwargs):
    """
    Checks whether extracted <answer> matches the normalized ground truth target.
    Normalizes both prediction and target using version_dict.

    Args:
        completions (list[str])
        targets (list[str])

    Returns:
        list[float]: Reward scores
    """
    rewards = []

    
    targets = kwargs['target']
        
    version_dict = {
        # English mappings
        "usa": "united states",
        "u.s.a.": "united states",
        "united states of america": "united states",
        "america": "united states",
        "bharat": "india",
        "hindustan": "india",
        "prc": "china",
        "people's republic of china": "china",
        "russian federation": "russia",
        "ksa": "saudi arabia",
        "kingdom of saudi arabia": "saudi arabia",
        "french republic": "france",
        "federal democratic republic of nepal": "nepal",
        "hellenic republic": "greece",
        "turkiye cumhuriyeti": "turkey",
        "republic of kenya": "kenya",
        "kingdom of thailand": "thailand",
        # Hindi mappings
        "अमेरिका": "संयुक्त राज्य अमेरिका",
        # you can expand more mappings
    }

    for completion, target in zip(completions, targets):
        try:
            completion = "<think>" + completion

            # Extract <answer>...</answer>
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue

            prediction = match.group(1).strip().lower()
            target_normalized = target.strip().lower()

            # Normalize both prediction and target
            prediction_normalized = version_dict.get(prediction, prediction)
            target_normalized = version_dict.get(target_normalized, target_normalized)

            if random.random() < 0.01:  # 1% chance to log predictions
                print(f"Prediction: {prediction_normalized}, Target: {target_normalized}")

            if prediction_normalized == target_normalized:
                rewards.append(1.0)
                
                # Log successful samples occasionally
                if random.random() < 0.10:  # 10% chance to log successful samples
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "successful_factual_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"Completion: {completion}\n")
                        f.write(f"Prediction: {prediction_normalized}, Target: {target_normalized}\n")
            else:
                rewards.append(0.0)

        except Exception as e:
            print(f"Error in reward function: {e}")
            rewards.append(0.0)

    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def prepare_factual_dataset(dataset_path, tokenizer):
    """
    Prepare the factual QA dataset for GRPO training
    """
    # Prepare prompt template for factual QA
    def generate_factual_prompt(question, answer):
        r1_prefix = [
            {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning behind the factual question using your internal knowledge in any language that can help you answer the given question. Think and reason in the language (for example, Hindi, Swahili, or any other language) if you feel that it can help you answer the question more accurately. Then, provide the final answer clearly in the same language as that of the question."},
            {"role": "user", "content": f"{question} Please think carefully and return your reasoning inside <think> </think> tags, and the final answer inside <answer> </answer> tags."},
            {"role": "assistant", "content": "Let me think step by step.\n<think>"}
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "target": answer
        }
    
    
    from datasets import Dataset
    import json

    # Open and load the JSON file
    with open('./data/factual_dataset.json', 'r') as f:
        data = json.load(f)           
    
    # Create a HuggingFace Dataset from all collected data
    train_dataset = Dataset.from_list(data['data'])

    logger.info(f"Loaded {len(train_dataset)} examples from HuggingFace dataset")
                    
    
    # Apply the prompt template to the dataset
    logger.info("Applying prompt template to dataset...")
    train_dataset = train_dataset.map(
        lambda x: generate_factual_prompt(x["question"], x["answers"])
    )
    
    # Split into train/test
    logger.info("Splitting dataset into train/test...")
    train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    
    return train_test_split["train"], train_test_split["test"]


def grpo_function(
    model_args, script_args, training_args
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Script arguments {script_args}")

    ################
    # Load tokenizer
    ################
    tokenizer_path = getattr(script_args, "tokenizer_name_or_path", None) or getattr(model_args, "model_name_or_path", None)
    model_revision = getattr(model_args, "model_revision", "main")
    trust_remote_code = getattr(model_args, "trust_remote_code", True)
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    #dataset_path = getattr(script_args, "dataset_id_or_path", None)
    dataset_path = "./data/factual_dataset"
    train_dataset, test_dataset = prepare_factual_dataset(
        dataset_path, tokenizer
    )

    logger.info(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    # Log a few examples from the dataset
    for i in range(min(3, len(train_dataset))):
        logger.info(f"Example {i}:")
        logger.info(f"Prompt: {train_dataset[i]['prompt'][:100]}...")
        logger.info(f"Target: {train_dataset[i]['target']}")

    #########################
    # Instantiate GRPO trainer
    #########################
    logger.info("Initializing GRPO trainer...")
    
    # Get necessary parameters from training_args
    output_dir = getattr(training_args, "output_dir", "runs/qwen-r1-factual-qa")
    logger.info(f"Using output directory: {output_dir}")
    
    # Create GRPO config from training_args
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=getattr(training_args, "learning_rate", 5e-7),
        lr_scheduler_type=getattr(training_args, "lr_scheduler_type", "cosine"),
        warmup_ratio=getattr(training_args, "warmup_ratio", 0.03),
        max_steps=getattr(training_args, "max_steps", 100),
        per_device_train_batch_size=getattr(training_args, "per_device_train_batch_size", 1),
        gradient_accumulation_steps=getattr(training_args, "gradient_accumulation_steps", 1),
        gradient_checkpointing=getattr(training_args, "gradient_checkpointing", True),
        logging_steps=getattr(training_args, "logging_steps", 10),
        save_steps=getattr(training_args, "save_steps", 25),
        max_prompt_length=getattr(training_args, "max_prompt_length", 256),
        max_completion_length=getattr(training_args, "max_completion_length", 1024),
        num_generations=getattr(training_args, "num_generations", 8),
        beta=getattr(training_args, "beta", 0.001),
        bf16=getattr(training_args, "bf16", True),
        use_vllm=getattr(training_args, "use_vllm", False),
    )
    
    # Create PEFT config
    peft_config = None
    if getattr(model_args, "use_peft", False):
        logger.info("Initializing PEFT config")
        peft_config = get_peft_config(model_args)
    
    trainer = GRPOTrainer(
        model=getattr(model_args, "model_name_or_path", "Qwen/Qwen2.5-3B-Instruct"),
        reward_funcs=[format_reward_func_factual, correctness_reward_func_factual],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
    )

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(grpo_config)
    if last_checkpoint is not None and getattr(grpo_config, "resume_from_checkpoint", None) is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ***'
    )
    #train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    train_result = trainer.train()
    
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    try:
        training_args.distributed_state.wait_for_everyone()  # wait for all processes to load
    except:
        logger.info("No distributed state found, continuing with model saving...")

    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")

    # Save everything else on main process
    try:
        if trainer.accelerator.is_main_process:
            trainer.create_model_card()
    except:
        logger.info("Creating model card failed, continuing with saving...")
        
    # push to hub if needed
    if getattr(training_args, "push_to_hub", False):
        logger.info("Pushing to hub...")
        try:
            trainer.push_to_hub()
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    
    model_args, script_args, training_args = parser.parse_args_and_config()
      
    # Set environment variables for distributed training
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Detected {torch.cuda.device_count()} GPUs. Enabling distributed training.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    # Enable TF32 for faster training on Ampere GPUs (A100)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 precision for faster training on A100 GPUs")

    # Enable TF32 for faster training on Ampere GPUs (A100) if supported
    if torch.cuda.is_available():
        try:
            # Check CUDA capabilities before enabling TF32
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 precision for faster training on A100 GPUs")
            else:
                logger.info(f"TF32 not enabled - GPU architecture ({major}.{minor}) does not support it")
        except Exception as e:
            logger.warning(f"Could not check CUDA capabilities: {e}")
            logger.info("TF32 precision not enabled")
    else:
        logger.info("CUDA not available, TF32 precision not enabled")

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()