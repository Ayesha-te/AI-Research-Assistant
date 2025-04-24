import streamlit as st
import toml
import sqlite3

from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import Tool
from langchain_community.tools.serpapi.tool import SerpAPIWrapper

# Load secrets from toml file
secrets = toml.load("secrets.toml")
openai_api_key = secrets["openai"]["apikey"]
serpapi_api_key = secrets["serpapi"]["apikey"]

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up LLM
llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Set up SerpAPI tool
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Useful for answering questions about current events or the world."
    )
]

# Initialize Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Streamlit App UI
st.set_page_config(page_title="AI Research Assistant")
st.title("🧠 AI Research Assistant")

# User input
query = st.text_input("Ask me anything:")

if query:
    response = agent.run(query)
    st.write("🤖", response)

    # Save to SQLite (optional feature)
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (question TEXT, answer TEXT)''')
    c.execute("INSERT INTO interactions (question, answer) VALUES (?, ?)", (query, response))
    conn.commit()
    conn.close()

    # Show past history
    with st.expander("📜 Conversation History"):
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "AI"
            st.markdown(f"**{role}:** {msg.content}")
