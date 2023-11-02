import os
import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup

# Set OpenAI API key using the SDK's dedicated method
OPENAI_KEY = st.secrets["openai"]["key"]

# Set up the Streamlit app
def main():
    st.title('ACN GPT Analyst')
    st.sidebar.image("ACN_LOGO.webp", caption='ACN', use_column_width=True)
    st.sidebar.title('African Collaborative Network(ACN)')

    # Initialize 'typed_query_history' session state if not present
    if 'typed_query_history' not in st.session_state:
        st.session_state.typed_query_history = []

    # Handle user queries
    user_query = st.text_input('Ask anything about Africans in the US:', ' ')

    if user_query:
        # Initialize a dictionary to store responses
        response_data = {"user_query": user_query, "responses": []}

        # Use OpenAI for generating responses
        response_obj = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_query}
            ]
        )
        response = response_obj['choices'][0]['message']['content']
        response_data["responses"].append({"name": "OpenAI", "response": response})

        # Display responses
        for source_data in response_data["responses"]:
            st.write(source_data['response'])

        # Store user query and response data in session state
        st.session_state.typed_query_history.append(response_data)

    # Display chat history with clickable buttons for typed queries
    st.sidebar.title('Typed Query History')
    clear_typed_query_history = st.sidebar.button("Clear Typed Query History")

    if clear_typed_query_history:
        st.session_state.typed_query_history = []

    for i, entry in enumerate(st.session_state.typed_query_history):
        query = entry["user_query"]
        for source_data in entry["responses"]:
            source_name = source_data["name"]
            source_response = source_data["response"]
            if st.sidebar.button(f"{i + 1}. {source_name}: {query}", key=f"typed_query_history_button_{i}_{source_name}"):
                st.write(f"Response for '{query}' from {source_name}:")
                st.write(source_response)

if __name__ == "__main__":
    main()
