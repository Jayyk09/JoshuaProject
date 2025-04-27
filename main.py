import pandas as pd
import os

def get_csv_shapes():
    data_dir = 'data'
    print("\nCSV File Shapes:")
    print("-" * 50)
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            try:
                # Read the first line to get database name
                with open(file_path, 'r') as f:
                    db_name = f.readline().strip()
                
                # Read the CSV starting from line 3 (skip first two lines)
                df = pd.read_csv(file_path, skiprows=2)
                print(f"Database: {db_name}")
                print(f"File: {file}")
                print(f"  Rows: {df.shape[0]}")
                print(f"  Columns: {df.shape[1]}")
                print(f"  Column names: {list(df.columns)}")
                print("-" * 50)
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")
                print("-" * 50)

if __name__ == "__main__":
    get_csv_shapes()
