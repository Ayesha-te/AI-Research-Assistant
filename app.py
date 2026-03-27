import sqlite3
from urllib.parse import urlencode

import requests
import streamlit as st
from openai import OpenAI


def get_client() -> OpenAI:
    return OpenAI(api_key=st.secrets["openai"]["apikey"])


def search_web(query: str, api_key: str) -> str:
    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "q": query,
            "api_key": api_key,
            "engine": "google",
            "num": 5,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    snippets = []
    for item in data.get("organic_results", [])[:5]:
        title = item.get("title", "Untitled")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        snippets.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}")

    return "\n\n".join(snippets) if snippets else "No web results were returned."


def generate_answer(question: str, web_context: str) -> str:
    prompt = (
        f"Question: {question}\n\n"
        f"Web research:\n{web_context}\n\n"
        "Answer the question using the research above. Be clear, practical, and mention uncertainty when needed."
    )
    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are a research assistant that synthesizes web findings."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


conn = sqlite3.connect("questions_responses.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS interactions (question TEXT, response TEXT)")
conn.commit()

st.title("AI Research Assistant")

user_input = st.text_input("Ask a question:")
if st.button("Submit"):
    if user_input:
        with st.spinner("Searching the web and preparing an answer..."):
            web_context = search_web(user_input, st.secrets["serpapi"]["apikey"])
            answer = generate_answer(user_input, web_context)

        st.write("Response:", answer)
        c.execute("INSERT INTO interactions (question, response) VALUES (?, ?)", (user_input, answer))
        conn.commit()

        with st.expander("Web research used"):
            st.text(web_context)

st.subheader("Previous Interactions")
for row in c.execute("SELECT * FROM interactions"):
    st.write(f"Q: {row[0]}")
    st.write(f"A: {row[1]}")
