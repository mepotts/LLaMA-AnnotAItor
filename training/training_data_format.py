#%%
import pandas as pd
import json
from typing import List, Dict
import re

def clean_text(text: str) -> str:
    """
    Clean and preprocess text
    - Remove extra whitespaces
    - Handle potential encoding issues
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', str(text)).strip()
    
    # Basic cleaning
    text = text.replace('\n', ' ').replace('\r', '')
    
    return text

def prepare_training_data(input_file: str, output_file: str):
    """
    Prepare training data in a format suitable for model fine-tuning
    """
    # Read the processed annotations
    df = pd.read_csv(input_file)
    
    # Prepare training samples
    training_samples: List[Dict] = []
    
    for _, row in df.iterrows():
        # Clean and prepare texts
        fragment = clean_text(row['fragment'])
        lyrics = clean_text(row['lyrics'])
        annotation = clean_text(row['annotation_comment'])
        
        # Create a structured prompt
        prompt = f"""
Analyze the following song lyric fragment:
Fragment: "{fragment}"
Context Lyrics: "{lyrics[:500]}"

Provide an insightful annotation that explains:
- Literary devices used
- Cultural or historical references
- Deeper meaning or interpretation
"""
        
        # Combine prompt and annotation
        training_sample = {
            "prompt": prompt.strip(),
            "completion": annotation.strip(),
            "metadata": {
                "votes": row['votes_total'],
                "song_id": row['song_id']
            }
        }
        
        training_samples.append(training_sample)
    
    # Save as JSON Lines format (common for LLM training)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    # Print some statistics
    print(f"Total training samples prepared: {len(training_samples)}")
    
    # Sample a few entries to verify
    print("\nSample Entries:")
    for sample in training_samples[:3]:
        print("\n--- Sample Entry ---")
        print("Prompt:", sample['prompt'][:300] + "...")
        print("Completion (preview):", sample['completion'][:200] + "...")

# Prepare training and validation data
prepare_training_data(
    '../data/processed/train_annotations.csv', 
    '../data/processed/train_data.jsonl'
)

prepare_training_data(
    '../data/processed/validation_annotations.csv', 
    '../data/processed/validation_data.jsonl'
)

# Quick verification of output files
import os
print("\nOutput Files:")
print("Training data size:", 
      os.path.getsize('../data/processed/train_data.jsonl') / (1024 * 1024), "MB")
print("Validation data size:", 
      os.path.getsize('../data/processed/validation_data.jsonl') / (1024 * 1024), "MB")