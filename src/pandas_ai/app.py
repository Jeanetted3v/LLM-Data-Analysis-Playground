"""To run:
chainlit run src/pandas_ai/app.py
"""
import pandas as pd
import chainlit as cl
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq 
import sqlite3
import os
print(f"Database exists: {os.path.exists('../test_data.db')}")

llm = ChatGroq(
    model_name="llama3-70b-8192",
    api_key = os.environ["GROQ_API_KEY"]
)

@cl.on_chat_start
def start_chat():
    # Set initial message history
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

@cl.on_message
async def main(message: cl.Message):
    # Retrieve message history
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    # Load data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    # excel_path = os.path.join(project_root, 'data/raw/test_data.xlsx')
    # df = pd.read_excel(excel_path)
    # df = pd.read_csv('data.csv')
    db_path = os.path.join(project_root, 'test_data.db')
    conn = sqlite3.connect(db_path)

    df = pd.read_sql('SELECT * FROM sheet', conn)
    conn.close()

    df = SmartDataframe(df, config={"llm": llm})
    
    question = message.content
    response = df.chat(question)
    msg = cl.Message(content=response)
    
    await msg.send()

    # Update message history and send final message
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    