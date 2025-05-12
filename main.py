import pandas as pd
import mysql.connector

from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("mysql+mysqlconnector://root:jay0912@localhost/joshuaproject")

# Load data from the table
df = pd.read_sql("SELECT * FROM jppeoples", engine)

# Check the primary key
primary_key_query = """
    SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'joshuaproject'
    AND TABLE_NAME = 'jppeoples'
    AND COLUMN_KEY = 'PRI'
"""
primary_key_df = pd.read_sql(primary_key_query, engine)
print("Primary Key(s):", primary_key_df['COLUMN_NAME'].tolist())

# Show sample
print(df.head())
print(f"Total rows: {len(df)}")
