import os 
import toml
import sqlite3
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from serpapi import GoogleSearch
#os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["apikey"]
#os.environ['SERPAPI_KEY']=st.secrets["serpapi"]["apikey"]
# Load secrets
tokens = toml.load("secrets.toml")
OPENAI_API_KEY = tokens["openai"]["api_key"]
SERPAPI_KEY = tokens["serpapi"]["api_key"]

# Setup memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Google Search tool
def search_google(query: str) -> str:
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY
    })
    results = search.get_dict()
    return results.get("organic_results", [{}])[0].get("snippet", "No result found.")

google_tool = Tool(
    name="GoogleSearch",
    func=search_google,
    description="Searches Google for recent information."
)

# Setup LLM
llm = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

# Setup agent
agent = initialize_agent(
    tools=[google_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Setup database
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

# CLI interaction
if __name__ == "__main__":
    while True:
        query = input("\nAsk a research question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = agent.run(query)
        print("\n🤖 Answer:", answer)
        log_conversation(query, answer)
