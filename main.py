import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from openai import OpenAI
import time
import re
from utils.prompt import prompt

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatabaseChatbot:
    def __init__(self, connection_string="mysql+mysqlconnector://root:jay0912@localhost/joshuaproject"):
        self.engine = create_engine(connection_string)
        self.schema_description = self.get_schema_description()
        self.messages = []
            

    def extract_sql_query(self, text):
        """Extract clean SQL query from the OpenAI response, removing any prefixes/code blocks."""
        # Remove 'sql' or 'SQL' at the beginning of the query
        text = text.strip()
        sql_prefix_pattern = re.compile(r'^sql\s*', re.IGNORECASE)
        text = sql_prefix_pattern.sub('', text)
        
        # Extract SQL from code blocks if present
        code_block_pattern = re.compile(r'```(?:sql)?(.*?)```', re.DOTALL)
        matches = code_block_pattern.findall(text)
        
        if matches:
            return matches[0].strip()
        
        return text.strip()
        

    def answer_question(self, question):
        """Answer a question about the database using conversation history for context"""        
        # Add the new question to conversation history
        self.messages.append({"role": "user", "content": question})
        
        # Step 1: Generate SQL query using OpenAI with conversation context
        try:
            # Prepare conversation context for SQL generation
            messages = prompt 
            
            # Add relevant conversation history (last 3 exchanges)
            for msg in self.messages[-6:]:
                messages.append(msg)
                
            # Add the current question with entity information
            messages.append({"role": "user", "content": f"Convert this question to SQL, using fuzzy matching where needed: {question}"})
            
            sql_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            # Extract the SQL query and clean it
            sql_query = self.extract_sql_query(sql_response.choices[0].message.content)
            print(f"Generated SQL: {sql_query}")
            
            # Add SQL query to conversation for context
            self.messages.append({"role": "assistant", "content": f"SQL: {sql_query}"})
            
        except Exception as e:
            error_msg = f"Error generating SQL query: {str(e)}"
            self.messages.append({"role": "assistant", "content": error_msg})
            return error_msg
        
        # Step 2: Execute the SQL query
        try:
            start_time = time.time()
            results_df = pd.read_sql(sql_query, self.engine)
            query_time = time.time() - start_time
            
            # Handle empty results
            if len(results_df) == 0:
                # Try a more general query based on geography
                for word, info in preprocessed["potential_entities"].items():
                    if info["entity_type"] == "geographic":
                        matches = info["matches"]
                        if isinstance(matches, list) and len(matches) > 0:
                            # Try to query using the first geographic match
                            location = matches[0]
                            fallback_query = f"""
                            SELECT PeopNameInCountry, Ctry, Population, PrimaryLanguageName, PrimaryReligion 
                            FROM jppeoples 
                            WHERE Ctry LIKE '%{location}%' OR ROG3 LIKE '%{location}%'
                            LIMIT 10
                            """
                            try:
                                fallback_df = pd.read_sql(fallback_query, self.engine)
                                if not fallback_df.empty:
                                    print(f"Fallback query found {len(fallback_df)} results")
                                    # Use the fallback results
                                    results_df = fallback_df
                                    break
                            except Exception as fallback_e:
                                print(f"Fallback query error: {fallback_e}")
                
                # If still no results, return no results message
                if len(results_df) == 0:
                    no_results_msg = "No matching data found in the database for your query."
                    self.conversation_history.append({"role": "assistant", "content": no_results_msg})
                    return no_results_msg
            
            # Convert to string with limited rows if large
            if len(results_df) > 10:
                data_str = results_df.head(10).to_string() + f"\n\n[Showing 10 of {len(results_df)} rows]"
            else:
                data_str = results_df.to_string()
                
            print(f"Query executed in {query_time:.2f} seconds, returned {len(results_df)} rows")
            
            # Add result summary to conversation history
            self.conversation_history.append({"role": "system", "content": f"Query returned {len(results_df)} rows"})
            
        except Exception as e:
            # Try to generate a better SQL query if there was an error
            try:
                print(f"SQL Error: {str(e)}")
                error_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are a helpful SQL assistant. Fix the SQL query that caused an error. Here is the schema:\n\n{self.schema_description}\n\nReturn ONLY the fixed SQL query with no explanations, prefixes, or code formatting. Do not include the word 'sql' at the beginning. Just return a valid SQL query that can be executed directly."},
                        {"role": "user", "content": f"This SQL query caused an error: {sql_query}\nError: {str(e)}\nPlease fix the query to answer the original question: {question}"}
                    ]
                )
                
                # Extract and clean the fixed SQL
                fixed_sql = self.extract_sql_query(error_response.choices[0].message.content)
                print(f"Fixed SQL: {fixed_sql}")
                
                results_df = pd.read_sql(fixed_sql, self.engine)
                if len(results_df) > 10:
                    data_str = results_df.head(10).to_string() + f"\n\n[Showing 10 of {len(results_df)} rows]"
                else:
                    data_str = results_df.to_string()
                    
                sql_query = fixed_sql  # Use the fixed query for the final response
                
                # Add fixed query to history
                self.conversation_history.append({"role": "system", "content": f"Fixed SQL: {fixed_sql}"})
                
            except Exception as nested_e:
                error_msg = f"Error executing SQL query: {str(e)}\nAttempted to fix, but got: {str(nested_e)}"
                self.conversation_history.append({"role": "assistant", "content": error_msg})
                return error_msg
        
        # Step 3: Send the results back to OpenAI for interpretation with conversation context
        try:
            # Prepare messages for the final interpretation
            interpret_messages = [
                {"role": "system", "content": "You are a helpful database assistant. Your task is to interpret database query results and provide a clear, accurate answer to the user's original question. Consider the conversation history for context and be conversational in your response."}
            ]
            
            # Add abbreviated conversation history (last 4 exchanges)
            for msg in self.conversation_history[-8:]:
                interpret_messages.append(msg)
                
            # Add the results
            interpret_messages.append({
                "role": "user", 
                "content": f"Based on our conversation and the latest results, please answer my question. Here are the query results:\n{data_str}"
            })
            
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=interpret_messages
            )
            
            answer = final_response.choices[0].message.content
            
            # Add the final answer to the conversation history
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            error_msg = f"Error interpreting results: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def get_conversation_summary(self):
        """Get a summary of the current conversation"""
        if len(self.conversation_history) <= 2:
            return "Conversation just started"
            
        try:
            summary_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize this conversation about a database in 1-2 sentences."},
                    {"role": "user", "content": str(self.conversation_history)}
                ]
            )
            return summary_response.choices[0].message.content
        except Exception as e:
            return f"Could not summarize conversation: {str(e)}"

def extract_sql_query(text):
    """Extract clean SQL query from the OpenAI response, removing any prefixes/code blocks."""
    # Remove 'sql' or 'SQL' at the beginning of the query
    text = text.strip()
    sql_prefix_pattern = re.compile(r'^sql\s*', re.IGNORECASE)
    text = sql_prefix_pattern.sub('', text)
    
    # Extract SQL from code blocks if present
    code_block_pattern = re.compile(r'```(?:sql)?(.*?)```', re.DOTALL)
    matches = code_block_pattern.findall(text)
    
    if matches:
        return matches[0].strip()
    
    return text.strip()

def answer_database_question(question, engine=None, schema_description=None):
    if engine is None:
        engine = create_engine("mysql+mysqlconnector://root:jay0912@localhost/joshuaproject")
    
    if schema_description is None:
        schema_description = get_schema_description(engine)
    
    # Step 1: Generate SQL query using OpenAI
    try:
        sql_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful SQL assistant. Your task is to convert a question about a database into a valid SQL query. The database has a table called 'jppeoples' with information about people groups worldwide. Here is the schema information:\n\n{schema_description}\n\nReturn ONLY the SQL query with no explanations, prefixes, or code formatting. Do not include the word 'sql' at the beginning of your response. Just return a valid SQL query that can be executed directly. Try string matching instead of directly putting the people group or language in the database into the query."},
                {"role": "user", "content": f"Convert this question to SQL: {question}"}
            ]
        )
        
        # Extract the SQL query and clean it
        sql_query = extract_sql_query(sql_response.choices[0].message.content)
        print(f"Generated SQL: {sql_query}")
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"
    
    # Step 2: Execute the SQL query
    try:
        start_time = time.time()
        results_df = pd.read_sql(sql_query, engine)
        query_time = time.time() - start_time
        
        # Handle empty results
        if len(results_df) == 0:
            return "No matching data found in the database for your query."
        
        # Convert to string with limited rows if large
        if len(results_df) > 10:
            data_str = results_df.head(10).to_string() + f"\n\n[Showing 10 of {len(results_df)} rows]"
        else:
            data_str = results_df.to_string()
            
        print(f"Query executed in {query_time:.2f} seconds, returned {len(results_df)} rows")
    except Exception as e:
        # Try to generate a better SQL query if there was an error
        try:
            print(f"SQL Error: {str(e)}")
            error_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a helpful SQL assistant. Fix the SQL query that caused an error. Here is the schema:\n\n{schema_description}\n\nReturn ONLY the fixed SQL query with no explanations, prefixes, or code formatting. Do not include the word 'sql' at the beginning. Just return a valid SQL query that can be executed directly."},
                    {"role": "user", "content": f"This SQL query caused an error: {sql_query}\nError: {str(e)}\nPlease fix the query to answer the original question: {question}"}
                ]
            )
            
            # Extract and clean the fixed SQL
            fixed_sql = extract_sql_query(error_response.choices[0].message.content)
            print(f"Fixed SQL: {fixed_sql}")
            
            results_df = pd.read_sql(fixed_sql, engine)
            if len(results_df) > 10:
                data_str = results_df.head(10).to_string() + f"\n\n[Showing 10 of {len(results_df)} rows]"
            else:
                data_str = results_df.to_string()
                
            sql_query = fixed_sql  # Use the fixed query for the final response
        except Exception as nested_e:
            return f"Error executing SQL query: {str(e)}\nAttempted to fix, but got: {str(nested_e)}"
    
    # Step 3: Send the results back to OpenAI for interpretation
    try:
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful database assistant. Your task is to interpret database query results and provide a clear, accurate answer to the user's original question."},
                {"role": "user", "content": f"Original question: {question}\nSQL Query: {sql_query}\nQuery results:\n{data_str}"}
            ]
        )
        
        return final_response.choices[0].message.content
    except Exception as e:
        return f"Error interpreting results: {str(e)}"

def get_schema_description(engine):
    """Get a description of the database schema to help OpenAI understand it better."""
    try:
        # Get column information
        columns_df = pd.read_sql("SHOW COLUMNS FROM jppeoples", engine)
        
        # Get a sample of data to understand types better
        sample_df = pd.read_sql("SELECT * FROM jppeoples LIMIT 3", engine)
        
        # Create schema description
        description = "Database Schema for jppeoples table:\n"
        for _, row in columns_df.iterrows():
            col_name = row['Field']
            col_type = row['Type']
            example_val = "NULL"
            
            if col_name in sample_df.columns:
                sample_val = sample_df[col_name].iloc[0]
                if pd.notna(sample_val):
                    example_val = f"{sample_val} (Example)"
            
            description += f"- {col_name}: {col_type}, Example: {example_val}\n"
            
        return description
    except Exception as e:
        print(f"Error getting schema: {e}")
        return "Could not retrieve schema information."

def run_interactive_mode():
    """Run an interactive CLI to ask questions about the database."""
    print("\nInitializing Database Chatbot...")
    chatbot = DatabaseChatbot()
    
    print("\n==== JoshuaProject Database AI Chatbot ====")
    print("Ask questions about the database or type 'exit' to quit.")
    print("Type 'clear' to start a new conversation.")
    print("Examples:")
    print("  - What are the top 5 largest people groups by population?")
    print("  - How many people groups are classified as 'Least Reached'?")
    print("  - Which countries have the highest percentage of evangelical Christians?")
    print("  - Tell me about Brahmins in India.")
    print("="*45)
    
    while True:
        question = input("\nYour question (or 'exit'/'clear'): ")
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        if question.lower() == 'clear':
            chatbot = DatabaseChatbot()  # Reset the chatbot
            print("\nConversation cleared. Starting fresh!")
            continue
            
        if not question.strip():
            continue
            
        print("\nProcessing your question...")
        answer = chatbot.answer_question(question)
        print("\nAnswer:")
        print("-" * 45)
        print(answer)
        print("-" * 45)
        
        # Show conversation summary after a few exchanges
        if len(chatbot.conversation_history) >= 6 and len(chatbot.conversation_history) % 2 == 0:
            print(f"\nConversation context: {chatbot.get_conversation_summary()}")

