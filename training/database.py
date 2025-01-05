#%%
import pyodbc
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Debug: Print environment variables
print(f"DB_SERVER: {os.getenv('DB_SERVER')}")
print(f"DB_DATABASE: {os.getenv('DB_DATABASE')}")

conn_str = (
    "DRIVER={SQL Server};"
    f"SERVER={os.getenv('DB_SERVER')};"
    f"DATABASE={os.getenv('DB_DATABASE')};"
    "Trusted_Connection=yes;"
)

# Debug: Print full connection string
print(f"Connection string: {conn_str}")

try:
    conn = pyodbc.connect(conn_str)
    print("Successfully connected to database!")
    conn.close()
except Exception as e:
    print(f"Error connecting to database: {e}")
# %%
