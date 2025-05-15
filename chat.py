import streamlit as st
import pandas as pd
import re
from main import DatabaseChatbot
from openai import OpenAI
import os

# Page config
st.set_page_config(
    page_title="JoshuaProject Database ChatBot",
    page_icon="🌍",
    layout="wide"
)

# Initialize the OpenAI client for fallback responses
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the chatbot
@st.cache_resource
def get_chatbot():
    return DatabaseChatbot()

# Function to handle geographic and entity fallbacks
def handle_fallback(query, chatbot, response):
    # If no results found in database
    if "No matching data found" in response:
        client = get_openai_client()
        
        # Extract country or region information
        country_match = re.search(r'(?:about|in|from)\s+(\w+)', query, re.IGNORECASE)
        potential_location = country_match.group(1) if country_match else None
        
        # Try to search by location instead
        if potential_location:
            # Try to get matching country data
            try:
                location_query = f"""
                SELECT PeopNameInCountry, Ctry, Population, PrimaryLanguageName, PrimaryReligion 
                FROM jppeoples 
                WHERE Ctry LIKE '%{potential_location}%' 
                OR ROG3 LIKE '%{potential_location}%'
                LIMIT 10
                """
                results_df = pd.read_sql(location_query, chatbot.engine)
                
                if not results_df.empty:
                    # Format location-based results
                    st.success(f"Found related people groups in {potential_location}:")
                    st.dataframe(results_df)
                    
                    # Generate an explanation of the results
                    context = f"""
                    The user asked about '{query}' but we didn't find exact matches.
                    Instead, we found {len(results_df)} people groups related to {potential_location}:
                    {results_df.to_string()}
                    
                    Provide a helpful summary of these people groups in {potential_location}.
                    """
                    
                    fallback_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You're a helpful assistant for information about people groups."},
                            {"role": "user", "content": context}
                        ]
                    )
                    
                    return True, fallback_response.choices[0].message.content
            except Exception as e:
                st.error(f"Error in geographic fallback: {str(e)}")
        
        # If no geographic fallback or it failed, provide general information
        try:
            context = f"""
            The user asked: '{query}'
            The database didn't have specific information about this. 
            Provide a helpful response about what the Joshua Project is and explain that 
            we might not have specific data about this particular query in our database.
            Suggest they try asking about specific people groups, countries, languages, or religions.
            """
            
            fallback_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You're a helpful assistant for the Joshua Project database."},
                    {"role": "user", "content": context}
                ]
            )
            
            return True, fallback_response.choices[0].message.content
        except Exception as e:
            st.error(f"Error in general fallback: {str(e)}")
    
    return False, response

# Main title
st.title("🌍 JoshuaProject Database ChatBot")
st.markdown("Ask questions about the JoshuaProject database to get insights about people groups worldwide.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "data" in message:
            # If there's dataframe data, display it
            st.dataframe(message["data"])
        st.markdown(message["content"])

# Display conversation summary if available
if len(st.session_state.messages) >= 6 and len(st.session_state.messages) % 2 == 0:
    chatbot = get_chatbot()
    with st.expander("Conversation Context"):
        st.info(chatbot.get_conversation_summary())

# Chat input
if prompt := st.chat_input("Ask a question about people groups worldwide..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get chatbot response
    chatbot = get_chatbot()
    
    # Display assistant response in chat
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            response = chatbot.answer_question(prompt)
            
            # Check if we need to handle fallback for no results
            used_fallback, response = handle_fallback(prompt, chatbot, response)
            
            # Check if response contains dataframe results (simplified detection)
            if not used_fallback and "[Showing" in response and "rows]" in response:
                try:
                    # Extract table data
                    table_start = response.find("\n")
                    table_end = response.find("\n\n[Showing")
                    
                    if table_start != -1 and table_end != -1:
                        table_text = response[table_start:table_end].strip()
                        response_text = response.replace(table_text, "").replace("\n\n", "\n")
                        
                        # Convert text table to dataframe (simplified approach)
                        # In a real app, better to modify the chatbot to return the dataframe
                        lines = table_text.strip().split("\n")
                        headers = lines[0].split()
                        data = []
                        for line in lines[1:]:
                            values = line.split()
                            if len(values) == len(headers):
                                data.append(values)
                        
                        if data:
                            df = pd.DataFrame(data, columns=headers)
                            st.dataframe(df)
                            
                        message_placeholder.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text, "data": df})
                    else:
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error parsing results: {str(e)}")
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("JoshuaProject Database AI Chatbot - Ask questions about people groups worldwide")

# Sidebar with example questions
with st.sidebar:
    st.subheader("Example Questions")
    examples = [
        "What are the top 5 largest people groups by population?",
        "How many people groups are classified as 'Least Reached'?",
        "Which countries have the highest percentage of evangelical Christians?",
        "Tell me about Brahmins in India.",
        "What languages do they speak?",
        "How many people groups practice Hinduism?"
    ]
    
    st.markdown("Try asking:")
    for example in examples:
        if st.button(example):
            # Use this example as the input
            st.session_state.messages.append({"role": "user", "content": example})
            
            # Get chatbot response
            chatbot = get_chatbot()
            response = chatbot.answer_question(example)
            
            # Check if we need to handle fallback for no results
            used_fallback, response = handle_fallback(example, chatbot, response)
            
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update the UI
            st.rerun()
    
    # Add a button to clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        chatbot = get_chatbot()
        chatbot.conversation_history = []
        st.rerun()
