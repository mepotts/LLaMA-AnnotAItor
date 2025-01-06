#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the filtered dataset
filtered_data = pd.read_csv('../data/filtered_annotations.csv')

# Function to create train/validation split
def create_train_validation_split(df, test_size=0.1, random_state=42):
    """
    Create train and validation splits with stratification
    
    Parameters:
    - df: Input DataFrame
    - test_size: Proportion of dataset to include in validation split
    - random_state: Seed for reproducibility
    
    Returns:
    - train_df: Training dataset
    - val_df: Validation dataset
    """
    # Group by song_id to ensure we don't leak annotations from same song across splits
    unique_songs = df['song_id'].unique()
    
    # Split unique songs
    train_songs, val_songs = train_test_split(
        unique_songs, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create train and validation DataFrames
    train_df = df[df['song_id'].isin(train_songs)]
    val_df = df[df['song_id'].isin(val_songs)]
    
    return train_df, val_df

# Perform the split
train_data, val_data = create_train_validation_split(filtered_data)

# Ensure directories exist
import os
os.makedirs('../data/processed', exist_ok=True)

# Save split datasets
train_data.to_csv('../data/processed/train_annotations.csv', index=False)
val_data.to_csv('../data/processed/validation_annotations.csv', index=False)

# Print split statistics
print("Dataset Split Statistics:")
print(f"Total Annotations: {len(filtered_data)}")
print(f"Training Annotations: {len(train_data)} ({len(train_data)/len(filtered_data)*100:.2f}%)")
print(f"Validation Annotations: {len(val_data)} ({len(val_data)/len(filtered_data)*100:.2f}%)")

print("\nTraining Set:")
print(f"Unique Songs: {train_data['song_id'].nunique()}")
print(f"Average Votes: {train_data['votes_total'].mean():.2f}")
print(f"Average Annotation Length: {train_data['annotation_length'].mean():.2f}")

print("\nValidation Set:")
print(f"Unique Songs: {val_data['song_id'].nunique()}")
print(f"Average Votes: {val_data['votes_total'].mean():.2f}")
print(f"Average Annotation Length: {val_data['annotation_length'].mean():.2f}")