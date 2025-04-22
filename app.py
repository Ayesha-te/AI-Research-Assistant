import os
import sqlite3
from datetime import datetime
import streamlit as st

# ✅ Updated LangChain imports for v0.1+
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import SerpAPIWrapper

from langchain.memory import ConversationBufferMemory

# ✅ Load API keys from Streamlit secrets
try:
    openai_api_key = st.secrets["openai"]["apikey"]
    serpapi_api_key = st.secrets["serpapi"]["apikey"]
except Exception as e:
    st.error("🚨 API keys not found in Streamlit secrets. Please add them.")
    st.stop()

# ✅ Set environment variables for APIs
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPAPI_API_KEY"] = serpapi_api_key

# ✅ Initialize SQLite database
def init_db():
    with sqlite3.connect('qa.db') as conn:
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

# ✅ Save Q&A to database
def save_to_db(question, answer):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect('qa.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO qa_log (question, answer, timestamp) VALUES (?, ?, ?)",
                  (question, answer, timestamp))
        conn.commit()

# ✅ Create agent (with tools and memory)
@st.cache_resource
def create_agent():
    llm = OpenAI(temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history")
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="Useful for answering questions with real-time Google Search."
        )
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

# ✅ Streamlit App UI
st.title("🧠 AI Research Assistant with Memory")
user_query = st.text_input("Ask me anything...")

# ✅ Initialize database and agent
init_db()
agent = create_agent()

# ✅ Handle user query
if user_query:
    response = agent.run(user_query)
    st.write("🧠", response)
    save_to_db(user_query, response)

# ✅ Show Q&A history
with st.expander("📜 Q&A History"):
    with sqlite3.connect('qa.db') as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM qa_log ORDER BY id DESC")
        rows = c.fetchall()
        for row in rows:
            st.write(f"🕒 {row[3]}")
            st.markdown(f"**Q:** {row[1]}")
            st.markdown(f"**A:** {row[2]}")
            st.markdown("---")

