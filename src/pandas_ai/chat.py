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
import tempfile
import time
import asyncio
from PIL import Image as PILImage
from matplotlib.figure import Figure
import uuid
import base64
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

charts_dir = os.path.join(tempfile.gettempdir(), "exports", "charts")
os.makedirs(charts_dir, exist_ok=True)

pai.config.set({
   "llm": llm,
   "temperature": 0,
   "seed": 26,
   "save_logs": True,
   "verbose": True,
   "max_retries": 3,
   "save_charts": True,
   "save_charts_path": charts_dir,
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
        logger.info(
            "CSV rows=%s, columns spec fields=%s",
            getattr(csv_df, "shape", ("?", "?")),
            len(columns or [])
        )

        # Build/Reuse PandasAI DataFrame ONCE per session
        pandasai_df = cl.user_session.get("pandasai_df")
        if pandasai_df is None:
            # Create the dataset (with schema) only once
            dataset = pai.create(
                path="my-org/companies",
                df=csv_df,
                description="Sales data from our retail stores",
                columns=columns,
                # force=True  # uncomment if you want to overwrite an existing registry entry
            )
            pandasai_df = pai.DataFrame(dataset, config={"llm": llm})
            cl.user_session.set("pandasai_df", pandasai_df)
            logger.info("PandasAI DataFrame created and cached in session.")
        else:
            logger.info("Reusing PandasAI DataFrame from session.")
        df = pandasai_df
        logger.info("PandasAI DataFrame created.")

        # Prompt
        message_history = cl.user_session.get("message_history")
        question = message.content
        full_prompt = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in message_history
        ) + f"\nUser: {question}"

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
            result = await asyncio.wait_for(loop.run_in_executor(None, _call_chat), timeout=60)
        except asyncio.TimeoutError:
            raise RuntimeError("Model call timed out after 60s. Check outbound internet / VPC egress / API quotas.")

        # --- Handle charts vs text ---
        to_send_text = None
        path = None

        def _save_and_send_pil(img: PILImage.Image):
            fname = f"chart_{uuid.uuid4().hex[:8]}.png"
            fpath = os.path.join(charts_dir, fname)
            img.save(fpath, format="PNG")
            return fpath

        def _save_and_send_mpl(fig: Figure):
            fname = f"chart_{uuid.uuid4().hex[:8]}.png"
            fpath = os.path.join(charts_dir, fname)
            fig.savefig(fpath, bbox_inches="tight")
            return fpath

        if isinstance(result, str):
            path = result

            # If PandasAI returned a relative path like "exports/charts/xyz.png",
            # normalize it to our /tmp-based charts_dir.
            if path.startswith("exports/charts/"):
                path = os.path.join(charts_dir, os.path.basename(path))

            # If it looks like an image and exists, render it
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")) and os.path.exists(path):
                await cl.Image(name=os.path.basename(path), path=path, display="inline").send()
            else:
                to_send_text = result or "(Empty response)"
        elif isinstance(result, PILImage.Image):
            path = _save_and_send_pil(result)
            await cl.Image(name=os.path.basename(path), path=path, display="inline").send()
        elif isinstance(result, Figure):
            path = _save_and_send_mpl(result)
            await cl.Image(name=os.path.basename(path), path=path, display="inline").send()
        # Sometimes libs return dicts like {"type":"image","path": "..."} or base64 data
        elif isinstance(result, dict):
            if "path" in result and isinstance(result["path"], str):
                path = result["path"]
                if path.startswith("exports/charts/"):
                    path = os.path.join(charts_dir, os.path.basename(path))
                if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")) and os.path.exists(path):
                    await cl.Image(name=os.path.basename(path), path=path, display="inline").send()
                else:
                    to_send_text = json.dumps(result)
            elif "image_base64" in result:
                fname = f"chart_{uuid.uuid4().hex[:8]}.png"
                path = os.path.join(charts_dir, fname)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(result["image_base64"]))
                await cl.Image(name=fname, path=path, display="inline").send()
            else:
                to_send_text = json.dumps(result)

        else:
            # Non-string results: stringify
            to_send_text = str(result) if result is not None else "(Empty response)"

        if to_send_text:
            await cl.Message(content=to_send_text).send()

        # Keep history consistent (use the same variable you used above)
        message_history.append({"role": "assistant", "content": to_send_text or os.path.basename(path)})
        cl.user_session.set("message_history", message_history)

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