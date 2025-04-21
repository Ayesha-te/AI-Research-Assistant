import os
import sqlite3
import streamlit as st

from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# --- Configuration: API Keys from Streamlit Secrets ---
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["apikey"]
os.environ["SERPAPI_API_KEY"] = st.secrets["serpapi"]["apikey"]

# --- Initialize SQLite Database ---
def init_db():
    conn = sqlite3.connect("qa_log.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS qa_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_qa(question, answer):
    conn = sqlite3.connect("qa_log.db")
    c = conn.cursor()
    c.execute("INSERT INTO qa_log (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

# --- Initialize LLM, Tools, Agent ---
llm = OpenAI(temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = load_tools(["serpapi"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🧠")
st.title("🧠 AI Research Assistant")
st.write("Ask a research question and I'll search the web to summarize the answer.")

# Initialize DB
init_db()

# Get user input
query = st.text_input("Enter your research question")

# If query submitted
if query:
    with st.spinner("Searching and summarizing..."):
        response = agent.run(query)
        save_qa(query, response)
        
        # Display Q&A
        st.markdown(f"**🧑 You:** {query}")
        st.markdown(f"**🤖 AI:** {response}")

# Show history
with st.expander("📜 Previous Questions & Answers"):
    conn = sqlite3.connect("qa_log.db")
    c = conn.cursor()
    c.execute("SELECT question, answer FROM qa_log ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    for question, answer in rows:
        st.markdown(f"**You:** {question}")
        st.markdown(f"**AI:** {answer}")
    conn.close()

# Show memory (debug)
with st.expander("🧠 Memory Log"):
    st.info(memory.buffer)

