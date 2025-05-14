"""To run:
chainlit run src/pandas_ai/app.py
Ref: https://mer.vin/2024/05/pandas-ai-database-excel-chainlit/
Ref: https://www.youtube.com/watch?v=p53YfWZJt14
"""
import pandas as pd
import chainlit as cl
from pandasai import SmartDataframe
import pandasai as pai
from pandasai_litellm import LiteLLM
import sqlite3
import os
from src.settings import SETTINGS


os.environ["OPENAI_API_KEY"] = SETTINGS.OPENAI_API_KEY
llm = LiteLLM(model="gpt-4.1-mini")
pai.config.set({
   "llm": llm,
   "temperature": 0,
   "seed": 26,
   "save_logs": True,
   "verbose": True,
   "max_retries": 3
})

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

    # for loading excel data
    # excel_path = os.path.join(project_root, 'data/raw/test_data.xlsx')
    # df = pd.read_excel(excel_path)

    # for loading csv data
    csv_path = os.path.join(project_root, 'data/processed/test_data.csv')
    df = pd.read_csv(csv_path)

    # for loading sql data
    # db_path = os.path.join(project_root, 'test_data_original.db')
    # conn = sqlite3.connect(db_path)

    # df = pd.read_sql('SELECT * FROM sheet', conn)
    # conn.close()

    df = SmartDataframe(df, config={"llm": llm})
    
    question = message.content
    full_prompt = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in message_history]
    )
    full_prompt += f"\nUser: {question}"
    response = df.chat(full_prompt)
    msg = cl.Message(content=response)
    
    await msg.send()

    # Update message history and send final message
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    