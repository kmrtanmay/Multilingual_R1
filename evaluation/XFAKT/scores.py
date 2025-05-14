#!/usr/bin/env python3
import random
import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats

# Define language groups
LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]

HIGH = ["English", "Chinese", "French", "Japanese"]
MEDIUM = ["Hindi", "Russian", "Arabic", "Greek", "Turkish"]
LOW = ["Nepali", "Ukrainian", "Swahili", "Thai"]

COUNTRIES = [
    "United States",
    "India",
    "China",
    "Russia",
    "Saudi Arabia",
    "France",
    "Nepal",
    "Japan",
    "Ukraine",
    "Greece",
    "Turkey",
    "Kenya",
    "Thailand"
]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calculate Cross-Lingual Factual Knowledge Transferability Score")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="factual_recall", help="Dataset type")
    parser.add_argument("--prompt", type=str, default="without_system_prompt", help="Prompt style")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Directory to save metrics results")
    return parser.parse_args()

def calculate_scores(model_path, dataset, prompt):
    """Calculate scores from evaluation results"""
    # Normalize model path if it's a directory
    model_name = os.path.split(model_path)[1] if os.path.exists(model_path) else model_path
    
    # Dictionary to store scores for each language
    data_scores = {}
    
    # Process each language
    for lang in LANGUAGES:
        # Get result paths
        result_path = f"results/{dataset}/{model_name}/{prompt}/{lang}.csv"
        english_result_path = f"results/{dataset}/{model_name}/{prompt}/English.csv"
        
        # Check if files exist
        if not os.path.exists(result_path):
            print(f"Warning: File not found - {result_path}")
            continue
            
        if not os.path.exists(english_result_path):
            print(f"Warning: English results file not found - {english_result_path}")
            continue
        
        try:
            # Load results
            data = pd.read_csv(result_path)
            data_english = pd.read_csv(english_result_path)
            
            # Initialize results and count
            results = {}
            count = {}
            
            # Process each row
            for i in range(len(data)):
                if i >= len(data_english):
                    continue
                    
                llm_output = data.iloc[i]["llm_output"]
                country = data_english.iloc[i]["answers"]
                
                # Initialize if not already done
                if country not in count:
                    count[country] = 0
                count[country] += 1
                
                if country not in results:
                    results[country] = 0
                
                # Check if output indicates incorrect answer ([4] marker)
                if "[4]" in llm_output:
                    results[country] += 1
            
            # Calculate error rates
            for key in results.keys():
                results[key] = round((results[key]/count[key]), 2) if count[key] > 0 else 0
            
            data_scores[lang] = results
            print(f"Processed {lang}: {len(data)} entries")
            
        except Exception as e:
            print(f"Error processing {lang}: {str(e)}")
    
    return data_scores

def calculate_metrics(data_scores):
    """Calculate cross-lingual metrics from scores"""
    # Prepare data arrays
    data_diagonal = []  # Same language-country pairs (diagonal)
    data_off_diagonal = []  # Different language-country pairs (off-diagonal)
    
    # Get available countries
    countries = list(next(iter(data_scores.values())).keys()) if data_scores else []
    
    # Fill arrays
    for lang in LANGUAGES:
        if lang not in data_scores:
            continue
            
        for country in COUNTRIES:
            if country not in data_scores[lang]:
                continue
                
            # Check if this is a diagonal pair (same language and country index)
            if LANGUAGES.index(lang) == COUNTRIES.index(country):
                data_diagonal.append(data_scores[lang][country])
            else:
                data_off_diagonal.append(data_scores[lang][country])
    
    # Convert to numpy arrays
    data_off_diagonal = np.array(data_off_diagonal)
    data_diagonal = np.array(data_diagonal)
    
    # Calculate metrics
    if len(data_diagonal) == 0 or len(data_off_diagonal) == 0:
        print("Not enough data to calculate metrics")
        return None, None, None
        
    mean_off = np.mean(data_off_diagonal)
    std_off = np.std(data_off_diagonal)
    mean_diagonal = np.mean(data_diagonal)
    std_diagonal = np.std(data_diagonal)
    
    # Run t-test
    try:
        t_stat, p_value = stats.ttest_ind(data_off_diagonal, data_diagonal)
    except:
        t_stat, p_value = 0, 1
    
    # Calculate factual recall score (FRS)
    frs = 1.5 * ((1 / (mean_off + mean_diagonal + 1)) - (1 / 3))
    
    # Calculate knowledge transfer score (KTS)
    kts = 2 * ((1 / (abs(mean_off - mean_diagonal) + 1)) - (1 / 2))
    
    # Calculate cross-lingual factual knowledge transferability score (X-FaKT)
    x_fakt = (2 * frs * kts) / (frs + kts) if (frs + kts) > 0 else 0
    
    return frs, kts, x_fakt

def calculate_language_group_metrics(data_scores):
    """Calculate metrics for each language group"""
    group_metrics = {}
    
    # Define language groups
    language_groups = {
        "HIGH": HIGH,
        "MEDIUM": MEDIUM,
        "LOW": LOW
    }
    
    # Calculate metrics for each group
    for group_name, languages in language_groups.items():
        # Filter scores for this group
        group_scores = {lang: scores for lang, scores in data_scores.items() if lang in languages}
        
        # Calculate metrics
        frs, kts, x_fakt = calculate_metrics(group_scores)
        
        if frs is not None:
            group_metrics[group_name] = {
                "FRS": frs,
                "KTS": kts,
                "X-FaKT": x_fakt
            }
    
    return group_metrics

def save_metrics(metrics, group_metrics, args):
    """Save metrics to CSV files"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save overall metrics
    overall_metrics_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_metrics.csv")
    
    with open(overall_metrics_path, "w") as f:
        f.write("Metric,Value\n")
        f.write(f"Factual Recall Score (FRS),{metrics['FRS']}\n")
        f.write(f"Knowledge Transfer Score (KTS),{metrics['KTS']}\n")
        f.write(f"Cross-Lingual Factual Knowledge Transferability Score (X-FaKT),{metrics['X-FaKT']}\n")
    
    # Save group metrics
    group_metrics_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_group_metrics.csv")
    
    with open(group_metrics_path, "w") as f:
        f.write("Group,FRS,KTS,X-FaKT\n")
        for group, metrics in group_metrics.items():
            f.write(f"{group},{metrics['FRS']},{metrics['KTS']},{metrics['X-FaKT']}\n")
    
    print(f"Metrics saved to {overall_metrics_path} and {group_metrics_path}")

def main():
    # Set random seed for reproducibility
    random.seed(0)
    
    # Parse arguments
    args = parse_args()
    
    print(f"Calculating metrics for model: {args.model}")
    
    # Calculate scores
    data_scores = calculate_scores(args.model, args.dataset, args.prompt)
    
    if not data_scores:
        print("Error: No valid data scores could be calculated.")
        return
    
    # Calculate overall metrics
    frs, kts, x_fakt = calculate_metrics(data_scores)
    
    if frs is None:
        print("Error: Could not calculate metrics.")
        return
    
    # Store metrics
    metrics = {
        "FRS": frs,
        "KTS": kts,
        "X-FaKT": x_fakt
    }
    
    # Calculate and store group metrics
    group_metrics = calculate_language_group_metrics(data_scores)
    
    # Print metrics
    print("\n=== Overall Metrics ===")
    print(f"Factual Recall Score: {frs:.4f}")
    print(f"Knowledge Transfer Score: {kts:.4f}")
    print(f"Cross-Lingual Factual Knowledge Transferability Score: {x_fakt:.4f}")
    
    print("\n=== Language Group Metrics ===")
    for group, metrics in group_metrics.items():
        print(f"{group}:")
        print(f"  FRS: {metrics['FRS']:.4f}")
        print(f"  KTS: {metrics['KTS']:.4f}")
        print(f"  X-FaKT: {metrics['X-FaKT']:.4f}")
    
    # Save metrics
    save_metrics(metrics, group_metrics, args)

if __name__ == "__main__":
    main()