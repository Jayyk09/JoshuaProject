import streamlit as st
import pandas as pd
import re
from main import DatabaseChatbot
import os
import random

# Page config
st.set_page_config(
    page_title="JoshuaProject Database ChatBot",
    page_icon="🌍",
    layout="wide"
)

# Initialize the chatbot
@st.cache_resource
def get_chatbot():
    try:
        return DatabaseChatbot()
    except Exception as e:
        st.error(f"Failed to initialize DatabaseChatbot: {e}")
        return None

# Function to extract dataframes from the response
def extract_dataframe_from_response(response_text):
    """Attempt to parse SQL results table from the agent response_text."""
    md_table_match = re.search(r"(\n|^)((?:\|.*\|\n)+)((?:\| ?-+:? ?-+\|.*\|\n)+)((?:(?:\|.*\|\n)+))(?:\n|$)", response_text, re.MULTILINE)
    if md_table_match:
        table_str = md_table_match.group(0).strip()
        try:
            lines = [line.strip() for line in table_str.split('\n') if line.strip()]
            if len(lines) > 1:
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                data_lines = [line for line in lines[2:] if '|' in line]
                data = []
                for line in data_lines:
                    values = [v.strip() for v in line.split('|') if v.strip()]
                    if len(values) == len(headers):
                        data.append(values)
                
                if data:
                    df = pd.DataFrame(data, columns=headers)
                    cleaned_response = response_text.replace(table_str, "").strip()
                    if not cleaned_response:
                        cleaned_response = "Here is the data I found:"
                    return True, cleaned_response, df
        except Exception as e:
            print(f"Error parsing markdown table: {e}")

    return False, response_text, None


# Main title
st.title("🌍 JoshuaProject Database ChatBot")
st.markdown("Ask questions about the JoshuaProject database to get insights about people groups worldwide.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message and message["data"] is not None:
            st.dataframe(message["data"])


# Chat input
if prompt := st.chat_input("Ask a question about people groups worldwide..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    chatbot = get_chatbot()
    
    if chatbot:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            df_placeholder = st.empty()

            with st.spinner("Thinking..."):
                raw_response = chatbot.chat(prompt)
                
                has_dataframe, final_text_response, df = extract_dataframe_from_response(raw_response)
                
                message_placeholder.markdown(final_text_response)
                if has_dataframe and df is not None:
                    df_placeholder.dataframe(df)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_text_response, 
                    "data": df
                })
    else:
        st.error("Chatbot is not available. Please check the console for errors during initialization.")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I am currently unavailable."})


# Footer
st.markdown("---")
st.markdown("JoshuaProject Database AI Chatbot")

# Sidebar with example questions
with st.sidebar:
    st.subheader("Example Questions")
    examples = [
        "What are the top 5 largest people groups by population?",
        "How many people groups are classified as 'Least Reached' in India?",
        "Which countries have the highest number of unreached people groups?",
        "Tell me about the Pashtun people in Afghanistan.",
        "What languages do the Berber people speak in Morocco?",
        "How many people groups practice Hinduism worldwide?",
        "List 5 people groups in Brazil."
    ]
    
    st.markdown("Try asking:")
    for example in examples:
        if st.button(example, key=random.randint(0, 1000000)):
            st.session_state.messages.append({"role": "user", "content": example})
            
            chatbot = get_chatbot()
            if chatbot:
                raw_response = chatbot.chat(example)
                has_dataframe, final_text_response, df = extract_dataframe_from_response(raw_response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_text_response, 
                    "data": df
                })
            else:
                 st.session_state.messages.append({"role": "assistant", "content": "Sorry, I am currently unavailable."})
            
            st.rerun()
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
