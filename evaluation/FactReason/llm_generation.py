import json
import torch
import re
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuned model on a test dataset")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test_dataset.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save result.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path")
    parser.add_argument("--lora_weights", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--override_chat_template", type=str, default=None, help="Override default chat template (if needed)")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs if available")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="Comma-separated list of GPU IDs to use (default: 0,1)")
    parser.add_argument("--no_triton", action="store_true", help="Disable dependencies on triton/bitsandbytes")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    print(f"Loading base model: {args.base_model}")
    
    # Set device map based on args
    device_map = None
    if args.multi_gpu:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            gpu_ids = [int(id) for id in args.gpu_ids.split(",")]
            print(f"Using multiple GPUs: {gpu_ids}")
            # Use 'auto' for automatic model sharding across available GPUs
            device_map = "auto"
        else:
            print("Multi-GPU requested but not available. Falling back to single GPU.")
    
    # Load base model with appropriate options
    model_kwargs = {
        "torch_dtype": torch.float16 if 'cuda' in args.device else torch.float32,
        "trust_remote_code": True,
    }
    
    # Add device_map only if using multi-GPU
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    
    print("Loading model with the following kwargs:", model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs
    )
    
    # If not using device_map, manually move model to specified device
    if device_map is None and 'cuda' in args.device:
        model = model.to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Enable chat templating for the tokenizer
    if args.override_chat_template:
        print(f"Overriding chat template with: {args.override_chat_template}")
        tokenizer.chat_template = args.override_chat_template
    elif not tokenizer.chat_template:
        print("Warning: Model does not have a default chat template. Using default Qwen chat template.")
        # Default Qwen chat template if not provided
        tokenizer.chat_template = "<|im_start|>user\n{{messages[0].content}}<|im_end|>\n<|im_start|>assistant\n"
    
    # Load LoRA weights with appropriate error handling
    print(f"Loading LoRA weights from: {args.lora_weights}")
    try:
        if args.no_triton:
            # Skip triton/bitsandbytes dependencies if flagged
            print("Skipping triton/bitsandbytes as requested")
            from peft import PeftConfig
            config = PeftConfig.from_pretrained(args.lora_weights)
            
            # For diagnostic purposes
            print(f"LoRA config: {config.__dict__}")
            
            # Try to load with safe settings
            model = PeftModel.from_pretrained(
                model, 
                args.lora_weights,
                is_trainable=False,
            )
        else:
            # Try with default settings
            model = PeftModel.from_pretrained(model, args.lora_weights)
    except ModuleNotFoundError as e:
        if "No module named 'triton.ops'" in str(e):
            print("Caught ModuleNotFoundError related to triton.ops")
            print("Trying to load LoRA without bitsandbytes/triton dependencies...")
            
            # Try alternative approach - load config first
            from peft import PeftConfig
            config = PeftConfig.from_pretrained(args.lora_weights)
            print(f"LoRA config loaded: {config.__dict__}")
            
            # Try to load with safe settings
            try:
                model = PeftModel.from_pretrained(
                    model, 
                    args.lora_weights,
                    is_trainable=False,
                )
            except Exception as inner_e:
                print(f"Still encountering error: {str(inner_e)}")
                print("Attempting one final approach...")
                
                # Try with config directly
                from peft import get_peft_model
                model = get_peft_model(model, config)
        else:
            # If it's a different error, re-raise it
            raise
    
    return model, tokenizer

def load_test_data(test_file):
    print(f"Loading test data from {test_file}")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data

def generate_reasoning(model, tokenizer, question, device, max_new_tokens=512, multi_gpu=False):
    # Use the specific template format provided
    messages = [
        {"role": "system", "content": "You are a helpful assistant. When answering a factual question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Think and reason carefully in the same language as the question (for example, if the question is in Hindi, then think and reason in Hindi).\n3. Consider multiple perspectives and potential answers before settling on your final response.\n4. Evaluate the confidence in your answer based on the information available to you.\n5. Provide the final answer clearly in the same language as the question, making sure it's well-supported by your reasoning.\n6. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."
        },
        {"role": "system", "content": "You are a helpful assistant. When answering a factual question, first think and reason in the same language as the question (for example, if question is in Hindi then think and reason in Hindi). Then, provide the final answer clearly in that same language."},
        {"role": "user", "content": f"{question} Please think carefully and return your reasoning inside <think> </think> tags, and the final direct answer inside <answer> </answer> tags."},
        {"role": "assistant", "content": "Let me think step by step.\n<think>"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True
    )
    #print(prompt)
    # Prepare input for model
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the right device(s)
    if not multi_gpu:
        # For single GPU, move inputs to that specific device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Lower temperature for more deterministic outputs
            do_sample=False,  # Greedy decoding
        )
    
    # Decode the response, removing the prompt
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Process the response to complete the <think> </think> structure if needed
    response = generated_text.strip()
    
    # If the model didn't complete the </think> tag and provide an <answer>, 
    # we need to handle it in post-processing
    if "</think>" not in response:
        response += "</think>"
    
    if "<answer>" not in response:
        response += "\n<answer>Unable to generate a complete answer</answer>"
    
    return response

def process_test_data(args, model, tokenizer, test_data):
    results = []
    model_name = f"{args.base_model}_lora_{args.lora_weights.split('/')[-1]}"
    
    print(f"Processing {len(test_data)} test examples...")
    for item in tqdm(test_data):
        question = item["question"]
        
        # Generate reasoning using the model
        llm_generation = generate_reasoning(
            model, 
            tokenizer, 
            question, 
            args.device, 
            args.max_new_tokens,
            args.multi_gpu
        )
        
        # Extract thinking and answer parts if possible
        think_content = ""
        answer_content = ""
        
        # Try to extract <think> content
        think_match = re.search(r'<think>(.*?)</think>', llm_generation, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        
        # Try to extract <answer> content
        answer_match = re.search(r'<answer>(.*?)</answer>', llm_generation, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        
        # Create result entry with original fields plus model's generation
        result = item.copy()
        result["llm_generation"] = llm_generation
        result["model_name"] = model_name
        
        # Add extracted reasoning and answer for easier analysis
        result["extracted_reasoning"] = think_content
        result["extracted_answer"] = answer_content
        
        results.append(result)
        
    return results

def save_results(results, output_file):
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    args = parse_args()
    
    # Set up GPU environment
    if args.multi_gpu and torch.cuda.is_available():
        gpu_ids = [int(id) for id in args.gpu_ids.split(",")]
        num_gpus = len(gpu_ids)
        
        if num_gpus > 1:
            # Set visible devices
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            print(f"Using {num_gpus} GPUs with IDs: {gpu_ids}")
        else:
            print(f"Only one GPU specified. Setting multi_gpu=False")
            args.multi_gpu = False
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Process test data
    results = process_test_data(args, model, tokenizer, test_data)
    
    # Save results
    save_results(results, args.output_file)
    
    print("Done!")

if __name__ == "__main__":
    import os
    main()