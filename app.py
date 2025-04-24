import os
import toml
import sqlite3
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper

# Load API keys from secrets.toml
secrets = toml.load("secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["openai"]["api_key"]
os.environ["SERPAPI_API_KEY"] = secrets["serpapi"]["api_key"]

# Streamlit app setup
st.set_page_config(page_title="AI Research Assistant")
st.title("🤖 AI Research Assistant")

# Memory for conversation
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Setup SerpAPI tool
search_tool = Tool(
    name="Google Search",
    func=SerpAPIWrapper().run,
    description="Useful for answering questions by searching the internet."
)

# Setup Agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state.memory,
    verbose=True
)

# Setup SQLite database
conn = sqlite3.connect("conversations.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

def log_conversation(question, response):
    cursor.execute("INSERT INTO chat_log (question, response) VALUES (?, ?)", (question, response))
    conn.commit()

# User input
query = st.text_input("Ask your research question:")

if query:
    response = agent.run(query)
    st.write("🧠", response)
    log_conversation(query, response)

    with st.expander("💬 Chat History"):
        for msg in st.session_state.memory.chat_memory.messages:
            role = "🧑 You" if msg.type == "human" else "🤖 AI"
            st.markdown(f"**{role}:** {msg.content}")
