"""First draft of POC

To run:
chainlit run src/pandas_ai/chat.py
Ref: https://mer.vin/2024/05/pandas-ai-database-excel-chainlit/
Ref: https://www.youtube.com/watch?v=p53YfWZJt14
"""
import logging
import pandas as pd
import json
import chainlit as cl
from pandasai import SmartDataframe
import pandasai as pai
from pandasai_litellm import LiteLLM
import os
import time
import asyncio
from google.cloud import storage
from src.utils.settings import SETTINGS
from hydra import compose, initialize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chainlit-app")

if not SETTINGS.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY missing from SETTINGS")
else:
    logger.info("OPENAI_API_KEY loaded (prefix): %s******", SETTINGS.OPENAI_API_KEY[:6])
os.environ["OPENAI_API_KEY"] = SETTINGS.OPENAI_API_KEY
llm = LiteLLM(model="gpt-4.1-mini", request_timeout=45)
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

BUCKET_NAME = "data_visualize_ai"
CSV_PATH = "test_data_shopname_lower.csv"
JSON_PATH = "test_data_columns.json"

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    logger.info("Chat session started.")
    logger.info(
        "Config bucket_name=%s csv_path=%s json_path=%s",
        getattr(cfg, "bucket_name", None),
        CSV_PATH,
        JSON_PATH,
    )

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})

    await cl.Message("Loading data from GCS…").send()
    logger.info("Loading data from GCS…")

    try:
        # GCS client + bucket
        storage_client = storage.Client()
        bucket_name = getattr(cfg, "bucket_name", None)
        if not bucket_name:
            raise RuntimeError("cfg.bucket_name is missing. Check config/config.yaml (bucket_name: ...).")
        logger.info("Using bucket: %s", bucket_name)
        bucket = storage_client.bucket(bucket_name)

        # Local tmp paths
        temp_csv_path = "/tmp/test_data.csv"
        temp_json_path = "/tmp/columns.json"

        # Blob existence + sizes
        csv_blob = bucket.blob(CSV_PATH)
        if not csv_blob.exists():
            raise FileNotFoundError(f"CSV not found: gs://{bucket_name}/{CSV_PATH}")
        logger.info("CSV exists. Size=%s bytes", csv_blob.size)

        json_blob = bucket.blob(JSON_PATH)
        if not json_blob.exists():
            raise FileNotFoundError(f"JSON not found: gs://{bucket_name}/{JSON_PATH}")
        logger.info("JSON exists. Size=%s bytes", json_blob.size)

        # Download
        t0 = time.time()
        csv_blob.download_to_filename(temp_csv_path)
        json_blob.download_to_filename(temp_json_path)
        logger.info("Downloaded files in %.2fs", time.time() - t0)

        await cl.Message("Parsing data…").send()
        csv_df = pai.read_csv(temp_csv_path)
        with open(temp_json_path, "r") as f:
            columns = json.load(f)
        logger.info("CSV rows=%s, columns spec fields=%s", getattr(csv_df, "shape", ("?", "?")), len(columns or []))

        # Build PandasAI DataFrame (keep your original flow)
        df = pai.create(
            path="my-org/companies",
            df=csv_df,
            description="Sales data from our retail stores",
            columns=columns,
        )
        df = pai.DataFrame(df, config={"llm": llm})
        logger.info("PandasAI DataFrame created.")

        # Prompt
        question = message.content
        full_prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history) + f"\nUser: {question}"
        await cl.Message("Calling the model…").send()
        logger.info("Calling df.chat()…")

        # Run df.chat in a worker with timeout so it can't hang forever
        loop = asyncio.get_running_loop()

        def _call_chat():
            t = time.time()
            try:
                return df.chat(full_prompt)
            finally:
                logger.info("df.chat() elapsed: %.2fs", time.time() - t)

        try:
            response = await asyncio.wait_for(loop.run_in_executor(None, _call_chat), timeout=60)
        except asyncio.TimeoutError:
            raise RuntimeError("Model call timed out after 60s. Check outbound internet / VPC egress / API quotas.")

        if not response:
            response = "(Empty response)"
        await cl.Message(content=response).send()

        history.append({"role": "assistant", "content": response})
        # Chainlit auto-update not needed here; sending once is fine.

    except Exception as e:
        logger.exception("Error in handler")
        err = (
            f"An error occurred: {e}\n\n"
            "Checks:\n"
            "• cfg.bucket_name set and GCS objects exist (we log sizes)\n"
            "• OPENAI_API_KEY set (prefix logged)\n"
            "• VPC egress not blocking internet (or enable Cloud NAT)\n"
            "• LiteLLM debug in logs shows HTTP activity to OpenAI\n"
        )
        await cl.Message(content=err).send()