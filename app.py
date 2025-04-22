# app.py

import os
import sqlite3
from datetime import datetime
import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from serpapi import GoogleSearch

# Load API keys from Streamlit secrets
openai_api_key = st.secrets["openai"]["apikey"]
serpapi_api_key = st.secrets["serpapi"]["apikey"]

# Set environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPAPI_API_KEY"] = serpapi_api_key

# SQLite setup
conn = sqlite3.connect('qa.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS qa_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        timestamp TEXT
    )
''')
conn.commit()

def save_to_db(question, answer):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO qa_log (question, answer, timestamp) VALUES (?, ?, ?)",
              (question, answer, timestamp))
    conn.commit()

# 🛠️ Custom SerpAPI Tool
def serpapi_search(query: str) -> str:
    search = GoogleSearch({
        "q": query,
        "api_key": serpapi_api_key,
    })
    results = search.get_dict()
    if "organic_results" in results:
        return "\n".join([res["title"] + ": " + res["link"] for res in results["organic_results"][:3]])
    else:
        return "No results found."

# Streamlit UI
st.title("🧠 AI Research Assistant (Manual SerpAPI)")
user_query = st.text_input("Ask me anything...")

# Define tools
tools = [
    Tool(
        name="Google Search",
        func=serpapi_search,
        description="Searches Google using SerpAPI for up-to-date information."
    )
]

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# LLM
llm = OpenAI(temperature=0.7)

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Run
if user_query:
    response = agent.run(user_query)
    st.write("🧠", response)
    save_to_db(user_query, response)

# Q&A History
with st.expander("📜 Q&A History"):
    c.execute("SELECT * FROM qa_log ORDER BY id DESC")
    rows = c.fetchall()
    for row in rows:
        st.write(f"🕒 {row[3]}")
        st.markdown(f"**Q:** {row[1]}")
        st.markdown(f"**A:** {row[2]}")
        st.markdown("---")
