import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from openai import OpenAI
import time
import re

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatabaseChatbot:
    def __init__(self, connection_string="mysql+mysqlconnector://root:jay0912@localhost/joshuaproject"):
        self.engine = create_engine(connection_string)
        self.schema_description = self.get_schema_description()
        self.conversation_history = []
        # Store commonly used entities for fuzzy matching
        self.cached_entities = {}
        self.initialize_entity_cache()
        
    def initialize_entity_cache(self):
        """Cache common entities from the database for fuzzy matching"""
        try:
            # Cache people groups
            people_groups = pd.read_sql("SELECT DISTINCT PeopNameInCountry FROM jppeoples", self.engine)
            self.cached_entities["people_groups"] = people_groups["PeopNameInCountry"].dropna().tolist()
            
            # Cache countries
            countries = pd.read_sql("SELECT DISTINCT Ctry FROM jppeoples", self.engine)
            self.cached_entities["countries"] = countries["Ctry"].dropna().tolist()
            
            # Cache languages
            languages = pd.read_sql("SELECT DISTINCT PrimaryLanguageName FROM jppeoples", self.engine)
            self.cached_entities["languages"] = languages["PrimaryLanguageName"].dropna().tolist()
            
            # Cache religions
            religions = pd.read_sql("SELECT DISTINCT PrimaryReligion FROM jppeoples", self.engine)
            self.cached_entities["religions"] = religions["PrimaryReligion"].dropna().tolist()
            
            print(f"Cached {len(self.cached_entities['people_groups'])} people groups, "
                  f"{len(self.cached_entities['countries'])} countries, "
                  f"{len(self.cached_entities['languages'])} languages, and "
                  f"{len(self.cached_entities['religions'])} religions for fuzzy matching.")
        except Exception as e:
            print(f"Warning: Could not initialize entity cache: {e}")

    def get_schema_description(self):
        """Get a description of the database schema to help OpenAI understand it better."""
        try:
            # Get column information
            columns_df = pd.read_sql("SHOW COLUMNS FROM jppeoples", self.engine)
            
            # Get a sample of data to understand types better
            sample_df = pd.read_sql("SELECT * FROM jppeoples LIMIT 3", self.engine)
            
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
    
    def find_fuzzy_matches(self, term, entity_type):
        """Find possible fuzzy matches for a term in the cached entities"""
        if entity_type not in self.cached_entities:
            return []
            
        matches = []
        term_lower = term.lower()
        
        # Check for exact match first
        for entity in self.cached_entities[entity_type]:
            if entity and term_lower == entity.lower():
                return [entity]  # Return exact match immediately
        
        # Simple fuzzy matching (contains or is contained by)
        for entity in self.cached_entities[entity_type]:
            if not entity:
                continue
                
            entity_lower = entity.lower()
            # Check if term is contained in entity or entity is contained in term
            if (term_lower in entity_lower or 
                entity_lower in term_lower or
                # Remove trailing 's' for plurals
                (term_lower.endswith('s') and term_lower[:-1] == entity_lower) or
                # Add trailing 's' for singular to plural
                (term_lower + 's' == entity_lower)):
                matches.append(entity)
                
        return matches[:5]  # Limit to top 5 matches
        
    def preprocess_question(self, question):
        """Pre-process question to handle entity mismatches"""
        # Extract potential entities from the question
        # This is a simple approach - in production you'd use NER
        words = re.findall(r'\b\w+\b', question)
        
        replacements = {}
        entity_types = ["people_groups", "countries", "languages", "religions"]
        
        for word in words:
            if len(word) < 4:  # Skip short words
                continue
                
            for entity_type in entity_types:
                matches = self.find_fuzzy_matches(word, entity_type)
                if matches:
                    replacements[word] = {
                        "matches": matches,
                        "entity_type": entity_type
                    }
                    break
        
        # Add fuzzy match info to the question context
        preprocessed = {
            "original_question": question,
            "potential_entities": replacements
        }
        
        return preprocessed

    def answer_question(self, question):
        """Answer a question about the database using conversation history for context"""
        # Pre-process question for fuzzy matching
        preprocessed = self.preprocess_question(question)
        
        # Add the new question to conversation history
        self.conversation_history.append({"role": "user", "content": question})
        
        # Step 1: Generate SQL query using OpenAI with conversation context
        try:
            # Prepare conversation context for SQL generation
            messages = [
                {"role": "system", "content": f"""You are a helpful SQL assistant. Your task is to convert a question about a database into a valid SQL query.
The database has a table called 'jppeoples' with information about people groups worldwide. 

Here is the schema information:
{self.schema_description}

Return ONLY the SQL query with no explanations, prefixes, or code formatting. 
Do not include the word 'sql' at the beginning of your response. 
Just return a valid SQL query that can be executed directly.

IMPORTANT: For fuzzy matching, use LIKE or similar operators. 
If the question mentions an entity that might not match exactly in the database, use pattern matching.
For example:
- Use '%Brahmin%' instead of 'Brahmin'
- Use '%Christian%' instead of 'Christian'

The user's question might contain entities that need fuzzy matching. Here are potential entities found:
{str(preprocessed['potential_entities'])}"""}
            ]
            
            # Add relevant conversation history (last 3 exchanges)
            for msg in self.conversation_history[-6:]:
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
            self.conversation_history.append({"role": "assistant", "content": f"SQL: {sql_query}"})
            
        except Exception as e:
            error_msg = f"Error generating SQL query: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
        
        # Step 2: Execute the SQL query
        try:
            start_time = time.time()
            results_df = pd.read_sql(sql_query, self.engine)
            query_time = time.time() - start_time
            
            # Handle empty results
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

def run_demo_mode():
    """Run a demonstration with predefined questions."""
    engine = create_engine("mysql+mysqlconnector://root:jay0912@localhost/joshuaproject")
    
    # Get schema info once at the beginning
    print("Analyzing database schema...")
    schema_description = get_schema_description(engine)
    
    # Show database schema
    print("Database Schema Summary:")
    columns_df = pd.read_sql("SHOW COLUMNS FROM jppeoples", engine)
    print(columns_df)
    
    # Demonstrate the function with sample questions
    sample_questions = [
        "What are the top 5 largest people groups by population?",
        "Which countries have the highest percentage of evangelical Christians?",
        "How many people groups are classified as 'Least Reached'?",
        "What is the distribution of primary religions across all people groups?"
    ]
    
    print("\n" + "="*50)
    print("DEMONSTRATING DATABASE Q&A FUNCTION")
    print("="*50)
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        answer = answer_database_question(question, engine, schema_description)
        print(f"Answer: {answer}")
        print("="*50)

def run_chatbot_demo():
    """Run a demonstration of the conversational chatbot with predefined questions."""
    print("\nInitializing Database Chatbot...")
    chatbot = DatabaseChatbot()
    
    # Demonstrate the chatbot with sample conversation
    sample_conversation = [
        "What are the top 5 largest people groups by population?",
        "Which of these are least reached?",
        "Tell me about Brahmins in India.",
        "What languages do they speak?",
        "How many people groups practice Hinduism?",
        "What is the percentage of evangelical Christians among these groups?"
    ]
    
    print("\n" + "="*50)
    print("DEMONSTRATING CONVERSATIONAL DATABASE CHATBOT")
    print("="*50)
    
    for i, question in enumerate(sample_conversation, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        answer = chatbot.answer_question(question)
        print(f"Answer: {answer}")
        print("="*50)
        
        # Show conversation context after a few exchanges
        if i >= 3:
            print(f"\nConversation context: {chatbot.get_conversation_summary()}")

if __name__ == "__main__":
    # Use command line args to control mode
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            run_demo_mode()
        elif sys.argv[1] == "--chatbot-demo":
            run_chatbot_demo()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options: --demo, --chatbot-demo")
    else:
        run_interactive_mode()