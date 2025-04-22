import os
import sqlite3
from datetime import datetime
import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from serpapi import GoogleSearch 


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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S
