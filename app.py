from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
import streamlit as st
import toml
import sqlite3

# Load secrets
secrets = toml.load("secrets.toml")
openai_api_key = secrets['openai']['api_key']
serpapi_api_key = secrets['serpapi']['api_key']

# Set up OpenAI and SerpAPI
llm = OpenAI(api_key=openai_api_key)
search_tool = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# Memory for context
memory = ConversationBufferMemory()

# LangChain agent
tools = [
    Tool(
        name="SerpAPI Search 🔍",
        func=search_tool.run,
        description="Search the web using SerpAPI"
    )
]

agent = initialize_agent(tools, llm, memory=memory, verbose=True)

# SQLite setup
conn = sqlite3.connect('questions_responses.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS interactions (question TEXT, response TEXT)''')
conn.commit()

# Streamlit app
st.title("AI Research Assistant 🤖✨")

user_input = st.text_input("Ask a question 📝:")
if st.button("Submit 🚀"):
    if user_input:
        response = agent.run(user_input)
        st.write("Response 🗣️:", response)
        c.execute("INSERT INTO interactions (question, response) VALUES (?, ?)", (user_input, response))
        conn.commit()

st.subheader("Previous Interactions 📚")
for row in c.execute("SELECT * FROM interactions"):
    st.write(f"Q: {row[0]} ❓")
    st.write(f"A: {row[1]} 💡")
