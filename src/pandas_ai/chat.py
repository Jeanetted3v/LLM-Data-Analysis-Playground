"""First draft of POC

To run:
chainlit run src/pandas_ai/chat.py
Ref: https://mer.vin/2024/05/pandas-ai-database-excel-chainlit/
Ref: https://www.youtube.com/watch?v=p53YfWZJt14
"""
import pandas as pd
import json
import chainlit as cl
from pandasai import SmartDataframe
import pandasai as pai
from pandasai_litellm import LiteLLM
import os
from google.cloud import storage
from src.utils.settings import SETTINGS
from hydra import compose, initialize


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

with initialize(version_base=None, config_path="../../config"):
    cfg = compose(config_name="config")

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

    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(cfg.buket_name)

        # Download CSV from GCS to temporary storage
        temp_csv_path = "/tmp/test_data.csv"
        temp_json_path = "/tmp/columns.json"

        # Download CSV from GCS to temporary storage
        csv_blob = bucket.blob(CSV_PATH)
        csv_blob.download_to_filename(temp_csv_path)
        
        # Download JSON from GCS to temporary storage
        json_blob = bucket.blob(JSON_PATH)
        json_blob.download_to_filename(temp_json_path)
        
        csv_file = pai.read_csv(temp_csv_path)
        with open(temp_json_path, 'r') as f:
                columns = json.load(f)

        df = pai.create(
            path="my-org/companies",
            df=csv_file,
            description="Sales data from our retail stores",
            columns=columns,
        )
        df = pai.DataFrame(df, config={"llm": llm})
        
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
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        msg = cl.Message(content=error_msg)
        await msg.send()
    