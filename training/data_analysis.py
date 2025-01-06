#%%
import pyodbc
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection
conn_str = (
    "DRIVER={SQL Server};"
    f"SERVER={os.getenv('DB_SERVER')};"
    f"DATABASE={os.getenv('DB_DATABASE')};"
    "Trusted_Connection=yes;"
)

# Connect to database
conn = pyodbc.connect(conn_str)

# Comprehensive data analysis query
query = """
WITH AnnotationStats AS (
    SELECT 
        COUNT(*) as total_annotations,
        AVG(CAST(votes_total AS FLOAT)) as avg_votes,
        STDEV(CAST(votes_total AS FLOAT)) as std_votes,
        MIN(votes_total) as min_votes,
        MAX(votes_total) as max_votes,
        
        AVG(LEN(fragment)) as avg_fragment_length,
        MIN(LEN(fragment)) as min_fragment_length,
        MAX(LEN(fragment)) as max_fragment_length,
        
        AVG(LEN(annotation_comment)) as avg_annotation_length,
        MIN(LEN(annotation_comment)) as min_annotation_length,
        MAX(LEN(annotation_comment)) as max_annotation_length,
        
        COUNT(DISTINCT song_id) as unique_songs,
        COUNT(DISTINCT annotator_id) as unique_annotators,
        
        SUM(CASE WHEN votes_total > 10 THEN 1 ELSE 0 END) as high_voted_annotations,
        SUM(CASE WHEN annotator_iq > 100000 THEN 1 ELSE 0 END) as high_iq_annotations
    FROM song_annotations
)
SELECT * FROM AnnotationStats
"""

# Execute query and load into pandas
df = pd.read_sql(query, conn)

# Display results
print("Data Quality Analysis:")
print(df.to_string(index=False))

# Additional analysis
def detailed_vote_distribution(conn):
    vote_query = """
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY votes_total) OVER () as votes_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY votes_total) OVER () as votes_p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY votes_total) OVER () as votes_p75,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY votes_total) OVER () as votes_p90
    FROM song_annotations
    """
    return pd.read_sql(vote_query, conn)

vote_dist = detailed_vote_distribution(conn)
print("\nVote Distribution Percentiles:")
print(vote_dist.to_string(index=False))

# Close connection
conn.close()