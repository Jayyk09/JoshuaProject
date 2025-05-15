import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import time  # Keep for potential future use (e.g., rate limiting)
import re    # Keep for potential future use (e.g., sanitizing output)
# LangChain specific imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent  # High-level agent constructor
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Ensure OPENAI_API_KEY is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

class DatabaseChatbot:
    def __init__(self, connection_string="mysql+mysqlconnector://root:jay0912@localhost/joshuaproject"):
        """
        Initializes the DatabaseChatbot.

        Args:
            connection_string (str): The SQLAlchemy connection string for the database.
        """
        print(f"Attempting to connect to database: {connection_string.split('@')[-1]}")
        try:
            self.engine = create_engine(connection_string)
            # Test connection
            with self.engine.connect() as connection:
                print("Successfully connected to the database.")
        except Exception as e:
            print(f"Failed to create database engine or connect: {e}")
            raise

        # 1. Wrap the engine with LangChain's SQLDatabase
        self.db = SQLDatabase(self.engine)

        # 2. Initialize an LLM
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key)

        # 3. Create the SQLDatabaseToolkit
        # The toolkit provides tools to the agent: sql_db_query, sql_db_schema, etc.
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        # 4. Create an agent that can use this toolkit
        # create_sql_agent is a convenient way to set up an agent for SQL tasks.
        # It automatically selects an appropriate prompt and agent type.
        # You can customize the agent_type (e.g., "openai-tools", "openai-functions", "zero-shot-react-description")
        # The dialect is important for the LLM to generate correct SQL.
        db_dialect = self.engine.dialect.name
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


        # Custom prompt to use LIKE for queries involving people names in countries
        custom_prompt_prefix = (
            f"You are an agent designed to interact with a {db_dialect} database. "
            "When asked about people names in a specific country, use the SQL LIKE operator to match patterns. "
            "Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.\n"
            "Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.\n"
            "You can order the results by a relevant column to return the most interesting examples in the database.\n"
            "Never query for all the columns from a specific table, only ask for the relevant columns given the question.\n"
            "You have access to tools for interacting with the database.\n"
            "Only use the given tools. Only use the information returned by the tools to construct your final answer.\n"
            "You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.\n"
            "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n"
            "If the question does not seem related to the database, just return \"I don't know\" as the answer."
        )

        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,  # Set to True to see the agent's thought process
            agent_type="openai-tools",  # "openai-tools" is generally recommended for newer OpenAI models
            prefix=custom_prompt_prefix,  # Use the custom prompt
            handle_parsing_errors=True,  # Handles errors if LLM output is not valid tool call
            memory=self.memory
        )
        print("DatabaseChatbot initialized successfully with agent.")

    def get_db_schema(self, tables=None):
        """
        Returns the schema for the specified tables, or all tables if None.
        """
        if tables:
            return self.db.get_table_info(table_names=tables.split(','))
        return self.db.get_table_info()

    def list_tables(self):
        """
        Returns a list of table names in the database.
        """
        return self.db.get_usable_table_names()

    def chat(self, user_query: str):
        """
        Takes a user's natural language query, uses the agent to get an answer from the database.

        Args:
            user_query (str): The user's question in natural language.

        Returns:
            str: The agent's answer.
        """
        if not self.agent_executor:
            return "Agent not initialized. Cannot process query."

        print(f"\nUser Query: {user_query}")
        try:
            # The input to the agent created by create_sql_agent is typically a dictionary with "input" key
            response = self.agent_executor.invoke({"input": user_query})
            # The actual answer is usually in the "output" key of the response dictionary
            answer = response.get("output", "No output found in agent response.")
            print(f"Agent Answer: {answer}")
            return answer
        except Exception as e:
            print(f"Error during agent execution: {e}")
            # You could try to extract more info if it's a LangChain specific error
            # For example, from an AgentExecutor's parsing error
            if hasattr(e, 'llm_output'):
                print(f"LLM Output was: {e.llm_output}")
            if hasattr(e, 'observation'):
                 print(f"Tool observation was: {e.observation}")
            return f"An error occurred: {str(e)}"

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Make sure your MySQL connection string is correct
        # pip install mysql-connector-python
        chatbot = DatabaseChatbot(connection_string="mysql+mysqlconnector://root:jay0912@localhost/joshuaproject")

        print("\n--- Listing Tables ---")
        tables = chatbot.list_tables()
        print(tables)

        if tables:  # Proceed only if tables are found
            print(f"\n--- Schema for table '{tables[0]}' ---")  # Get schema for the first table
            # schema = chatbot.get_db_schema(tables=tables[0]) # If only one table
            schema = chatbot.get_db_schema()  # Get schema for all tables
            print(schema)

            print("\n--- Chatting with the Database ---")
            query1 = "what percentage of chinese are evangelical Christian?"
            answer1 = chatbot.chat(query1)
            print(f"\nQ: {query1}\nA: {answer1}")

        else:
            print("No tables found in the database. Cannot proceed with chat examples.")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")