import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Example usage
file_path = 'data/AllLanguageListing.csv'
number_of_rows = get_number_of_rows(file_path)
print(f"Number of rows in {file_path}: {number_of_rows}")


# Retrieve Supabase URL and Key from environment variables
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(url)

# Initialize Supabase client
supabase: Client = create_client(url, key)

# Query the "LanguageListing" table
response = (supabase.table("LanguageListing").select("*").execute())

# Check if the query was successful
print(len(response.data))


