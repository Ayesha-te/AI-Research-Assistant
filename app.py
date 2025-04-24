import os
import toml
import sqlite3
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SerpAPIWrapper
import streamlit as st

# Load secrets
secrets = toml.load("secrets.toml")
openai_api_key = secrets['openai']['api_key']
serpapi_api_key = secrets['serpapi']['api_key']

# Set up OpenAI and SerpAPI
llm = OpenAI(api_key=openai_api_key)
search_tool = SerpAPIWrapper(api_key=serpapi_api_key)

# Set up memory for context-aware responses
memory = ConversationBufferMemory()

# Initialize LangChain agent
tools = [
    Tool(
        name="SerpAPI Search",
        func=search_tool.search,
        description="Use this tool to search the web using SerpAPI."
    )
]

agent = initialize_agent(tools, llm, memory=memory, verbose=True)

# Set up SQLite database
conn = sqlite3.connect('questions_responses.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS interactions
             (question TEXT, response TEXT)''')
conn.commit()

# Streamlit app
st.title("AI Research Assistant")

user_input = st.text_input("Ask a question:")
if st.button("Submit"):
    if user_input:
        response = agent.run(user_input)
        st.write("Response:", response)

        # Store question and response in the database
        c.execute("INSERT INTO interactions (question, response) VALUES (?, ?)", (user_input, response))
        conn.commit()

# Display previous interactions
st.subheader("Previous Interactions")
for row in c.execute("SELECT * FROM interactions"):
    st.write(f"Q: {row[0]}")
    st.write(f"A: {row[1]}")

