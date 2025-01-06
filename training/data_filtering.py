#%%
import pyodbc
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure data directory exists
os.makedirs('../data', exist_ok=True)

# Database connection
conn_str = (
    "DRIVER={SQL Server};"
    f"SERVER={os.getenv('DB_SERVER')};"
    f"DATABASE={os.getenv('DB_DATABASE')};"
    "Trusted_Connection=yes;"
)

# Quality Filtering Query
def create_filtered_dataset():
    query = """
    WITH FilteredAnnotations AS (
        SELECT 
            sa.*,
            sl.lyrics,
            -- Basic quality filters
            CASE 
                WHEN 
                    -- Positive or meaningful votes
                    sa.votes_total > 3 
                    -- High-quality annotator
                    OR (sa.annotator_iq > 100000 AND sa.votes_total >= 0)
                THEN 1 
                ELSE 0 
            END as quality_score,
            
            -- Additional filtering criteria
            LEN(sa.fragment) AS fragment_length,
            LEN(sa.annotation_comment) AS annotation_length
        FROM 
            song_annotations sa
        JOIN 
            song_lyrics sl ON sa.song_id = sl.song_id
        WHERE 
            -- Remove extreme outliers and nonsensical annotations
            sa.fragment IS NOT NULL 
            AND sa.annotation_comment IS NOT NULL
            AND LEN(sa.fragment) BETWEEN 10 AND 500  -- Reasonable fragment length
            AND LEN(sa.annotation_comment) BETWEEN 50 AND 2000  -- Meaningful annotation length
            AND sa.votes_total >= 0  -- Non-negative votes
    )
    SELECT 
        song_id,
        fragment,
        annotation_comment,
        lyrics,
        votes_total,
        annotator_iq,
        quality_score,
        fragment_length,
        annotation_length
    FROM 
        FilteredAnnotations
    WHERE 
        quality_score = 1  -- Only high-quality annotations
    ORDER BY 
        votes_total DESC  -- Prioritize most voted annotations
    """
    
    # Connect and execute
    conn = pyodbc.connect(conn_str)
    filtered_df = pd.read_sql(query, conn)
    conn.close()
    
    return filtered_df

# Execute filtering
filtered_data = create_filtered_dataset()

# Analyze filtered dataset
print("Filtered Dataset Statistics:")
print(f"Total Annotations: {len(filtered_data)}")
print(f"Unique Songs: {filtered_data['song_id'].nunique()}")

# Basic distribution checks
print("\nVote Distribution:")
print(filtered_data['votes_total'].describe())

print("\nAnnotation Length Distribution:")
print(filtered_data['annotation_length'].describe())

# Optional: Save filtered dataset
filtered_data.to_csv('../data/filtered_annotations.csv', index=False)
print("\nFiltered dataset saved to ../data/filtered_annotations.csv")