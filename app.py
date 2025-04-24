import os
import toml
import sqlite3
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SerpAPIWrapper


openai_api_key = st.secrets["openai"]["apikey"]
serpapi_api_key = st.secrets["serpapi"]["apikey"]

# Set up OpenAI LLM
llm = OpenAI(openai_api_key=openai_api_key)

# Set up SerpAPI
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# Create LangChain memory
memory = ConversationBufferMemory()

# Define tools
tools = [
    Tool(
        name="SerpAPI Search 🔍",
        func=search.run,
        description="Use this tool to search the web using SerpAPI."
    )
]

# Initialize agent
agent = initialize_agent(tools, llm, memory=memory, verbose=True)

# SQLite setup
conn = sqlite3.connect('questions_responses.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS interactions (question TEXT, response TEXT)''')
conn.commit()

# Streamlit UI
st.title("AI Research Assistant 🤖✨")

user_input = st.text_input("Ask a question 📝:")
if st.button("Submit 🚀"):
    if user_input:
        response = agent.run(user_input)
        st.write("Response 🗣️:", response)

        # Save to DB
        c.execute("INSERT INTO interactions (question, response) VALUES (?, ?)", (user_input, response))
        conn.commit()

# Show history
st.subheader("Previous Interactions 📚")
for row in c.execute("SELECT * FROM interactions"):
    st.write(f"Q: {row[0]} ❓")
    st.write(f"A: {row[1]} 💡")
