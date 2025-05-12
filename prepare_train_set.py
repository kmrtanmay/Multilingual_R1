import pandas as pd
import json
import os
from tqdm import tqdm

def load_csv(file_path="./factual.csv"):
    """
    Load the CSV file containing multilingual questions with translations.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with questions and translations
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def merge_answer_lists(original_list_json, translated_list_json):
    """
    Merge original and translated answer lists, removing duplicates.
    
    Args:
        original_list_json (str): JSON string of original answers
        translated_list_json (str): JSON string of translated answers
        
    Returns:
        list: Combined list of unique answers
    """
    # Parse JSON strings to lists
    try:
        original_list = json.loads(original_list_json) if original_list_json and pd.notna(original_list_json) else []
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse original answer list: {original_list_json}")
        original_list = []
    
    try:
        translated_list = json.loads(translated_list_json) if translated_list_json and pd.notna(translated_list_json) else []
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse translated answer list: {translated_list_json}")
        translated_list = []
    
    # Combine lists and remove duplicates while preserving order
    combined = []
    for item in original_list + translated_list:
        if item and item not in combined:
            combined.append(item)
    
    return combined

def create_language_entry(question_id, question, answers, region, topic, answer_type, lang_code):
    """
    Create a single language-specific entry for the JSONL file.
    
    Args:
        question_id (str): Unique identifier for the question
        question (str): Question text in the specific language
        answers (list): List of possible answers in the specific language
        region (str): Region the question is about
        topic (str): Topic of the question
        answer_type (str): Type of answer expected
        lang_code (str): Language code (en, hi, ja, sw, th)
        
    Returns:
        dict: Entry dictionary ready for JSON serialization
    """
    # Select first answer as main answer if available, otherwise empty string
    main_answer = answers[0] if answers else ""
    
    # Create the entry
    entry = {
        "id": f"{question_id}_{lang_code}",
        "question": question if pd.notna(question) else "",
        "answers": main_answer,
        "answer_list": answers,
        "region": region if pd.notna(region) else "",
        "topic": topic if pd.notna(topic) else "",
        "answer_type": answer_type if pd.notna(answer_type) else "",
        "language": lang_code
    }
    
    return entry

def convert_to_jsonl(df, output_file="./multilingual_questions.jsonl"):
    """
    Convert the DataFrame to JSONL format, creating separate entries for each language.
    
    Args:
        df (pandas.DataFrame): DataFrame with multilingual questions and translations
        output_file (str): Path to the output JSONL file
        
    Returns:
        int: Number of entries written to the JSONL file
    """
    # Language codes to process
    languages = ['en', 'hi', 'ja', 'sw', 'th']
    
    # Count of entries written
    entry_count = 0
    
    # Open the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Process each row
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to JSONL"):
            # Extract common fields
            question_id = str(row.get('question_id', ''))
            region = row.get('region', '')
            topic = row.get('topic', '')
            answer_type = row.get('answer_type', '')
            
            # Process each language
            for lang in languages:
                # Get question for this language
                question = row.get(f'question_{lang}', '')
                
                # Skip if no question for this language
                if not pd.notna(question) or not question.strip():
                    continue
                
                # Get original answers for this language
                answers_json = row.get(f'answers_{lang}', '[]')
                
                # Get translated answers for this language (except English)
                if lang != 'en':
                    translated_answers_json = row.get(f'answers_translate_{lang}', '[]')
                    
                    # Merge original and translated answers
                    answers = merge_answer_lists(answers_json, translated_answers_json)
                else:
                    # For English, use only the original answers
                    try:
                        answers = json.loads(answers_json) if pd.notna(answers_json) else []
                    except (json.JSONDecodeError, TypeError):
                        print(f"Warning: Could not parse English answer list: {answers_json}")
                        answers = []
                
                # Create entry for this language
                entry = create_language_entry(
                    question_id, question, answers, region, topic, answer_type, lang
                )
                
                # Write entry to JSONL file
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entry_count += 1
    
    print(f"Successfully wrote {entry_count} entries to {output_file}")
    return entry_count

def main(input_csv="./factual.csv", output_jsonl="./multilingual_questions.jsonl"):
    """
    Main function to convert CSV to JSONL.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_jsonl (str): Path to the output JSONL file
        
    Returns:
        int: Number of entries written or None if error
    """
    # Load CSV
    df = load_csv(input_csv)
    if df is None:
        return None
    
    # Check for required columns
    required_prefixes = ['question_', 'answers_']
    language_codes = ['en', 'hi', 'ja', 'sw', 'th']
    
    missing_columns = []
    for prefix in required_prefixes:
        for lang in language_codes:
            col = f"{prefix}{lang}"
            if col not in df.columns:
                missing_columns.append(col)
    
    if missing_columns:
        print(f"Warning: Some expected columns are missing: {missing_columns}")
        print("Will proceed with available columns only.")
    
    # Convert to JSONL
    entry_count = convert_to_jsonl(df, output_jsonl)
    
    return entry_count

# Display a sample of the JSONL file for verification
def show_jsonl_sample(jsonl_file="./multilingual_questions.jsonl", sample_size=3):
    """
    Display a sample of entries from the JSONL file.
    
    Args:
        jsonl_file (str): Path to the JSONL file
        sample_size (int): Number of entries to display
    """
    if not os.path.exists(jsonl_file):
        print(f"File not found: {jsonl_file}")
        return
    
    print(f"\nSample entries from {jsonl_file}:")
    print("="*50)
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
                
            try:
                entry = json.loads(line)
                print(f"Entry {i+1}:")
                print(f"ID: {entry.get('id')}")
                print(f"Language: {entry.get('language')}")
                print(f"Question: {entry.get('question')}")
                print(f"Main Answer: {entry.get('answers')}")
                print(f"Answer List: {entry.get('answer_list')}")
                print(f"Topic: {entry.get('topic')}, Type: {entry.get('answer_type')}, Region: {entry.get('region')}")
                print("-"*50)
            except json.JSONDecodeError:
                print(f"Error parsing line {i+1}")
    
    print("="*50)

if __name__ == "__main__":
    # Convert CSV to JSONL
    entry_count = main()
    
    # Show sample of the output
    if entry_count:
        show_jsonl_sample()
else:
    # This will help when importing the module in Jupyter
    print("Import successful. Call main() to convert CSV to JSONL.")