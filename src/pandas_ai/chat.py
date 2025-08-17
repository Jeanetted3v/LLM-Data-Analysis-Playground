"""Fixed image handling for GCP deployment

Key changes:
1. Convert images to base64 content for reliable display
2. Upload charts back to GCS for persistent storage
3. Use content bytes instead of file paths
"""

import logging
import json
import chainlit as cl
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
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chainlit-app")

# Initialize OpenAI API key
if not SETTINGS.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY missing from SETTINGS")
else:
    logger.info("OPENAI_API_KEY loaded (prefix): %s******", SETTINGS.OPENAI_API_KEY[:6])
os.environ["OPENAI_API_KEY"] = SETTINGS.OPENAI_API_KEY
llm = LiteLLM(model="gpt-4.1-mini", request_timeout=45)

# Create charts directory
charts_dir = os.path.join(tempfile.gettempdir(), "exports", "charts")
os.makedirs(charts_dir, exist_ok=True)

# Configure PandasAI
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

logger.info("PandasAI config - charts_dir: %s", charts_dir)
logger.info("Current working directory: %s", os.getcwd())
logger.info("Charts directory exists: %s", os.path.exists(charts_dir))

# Load Hydra configuration
with initialize(version_base=None, config_path="../../config"):
    cfg = compose(config_name="config")

BUCKET_NAME = "data_visualize_ai"
CSV_PATH = "test_data_shopname_lower.csv"
JSON_PATH = "test_data_columns.json"

def _convert_image_to_base64_content(image_path_or_object):
    """Convert image to base64 content bytes for reliable display"""
    try:
        if isinstance(image_path_or_object, str):
            # Handle file path
            with open(image_path_or_object, "rb") as f:
                return f.read()
        elif isinstance(image_path_or_object, PILImage.Image):
            # Handle PIL Image
            img_buffer = io.BytesIO()
            image_path_or_object.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
        elif isinstance(image_path_or_object, Figure):
            # Handle matplotlib Figure
            img_buffer = io.BytesIO()
            image_path_or_object.savefig(img_buffer, format='png', bbox_inches='tight')
            return img_buffer.getvalue()
    except Exception as e:
        logger.error("Failed to convert image to content: %s", e)
        return None

async def _upload_chart_to_gcs(content_bytes, bucket, filename_prefix="chart"):
    """Upload chart to GCS for persistent storage"""
    try:
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
        blob = bucket.blob(f"generated_charts/{filename}")
        blob.upload_from_string(content_bytes, content_type='image/png')
        
        # Make blob publicly readable (optional, for direct URL access)
        blob.make_public()
        
        logger.info("Uploaded chart to GCS: %s", filename)
        return blob.public_url, filename
    except Exception as e:
        logger.error("Failed to upload chart to GCS: %s", e)
        return None, None

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

        # ... (keep your existing data loading code) ...
        
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
            dataset = pai.create(
                path="my-org/companies",
                df=csv_df,
                description="Sales data from our retail stores",
                columns=columns,
            )
            pandasai_df = pai.DataFrame(dataset, config={"llm": llm})
            cl.user_session.set("pandasai_df", pandasai_df)
            logger.info("PandasAI DataFrame created and cached in session.")
        else:
            logger.info("Reusing PandasAI DataFrame from session.")
        df = pandasai_df

        # Prompt
        message_history = cl.user_session.get("message_history")
        question = message.content
        full_prompt = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in message_history
        ) + f"\nUser: {question}"

        await cl.Message("Calling the model…").send()
        logger.info("Calling df.chat()…")

        # Run df.chat in a worker with timeout
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

        logger.info("Result type: %s, content: %s", type(result).__name__, str(result)[:200])
        
        # Debug: Check what attributes the result object has
        if hasattr(result, '__dict__'):
            logger.info("Result attributes: %s", list(vars(result).keys()))
        if hasattr(result, '__class__'):
            logger.info("Result class: %s", result.__class__.__name__)

        # --- IMPROVED CHART HANDLING FOR GCP DEPLOYMENT ---
        to_send_text = None
        image_sent = False

        # Handle different result types with content-based approach
        # First check if it's a ChartResponse object or similar
        if (hasattr(result, 'value') and hasattr(result, 'image')) or \
           (hasattr(result, 'chart') or 'chart' in str(type(result)).lower() or 'response' in str(type(result)).lower()):
            # This is likely a ChartResponse or similar object
            logger.info("Handling ChartResponse-like object")
            
            # Method 1: Try to access the image attribute directly
            image_obj = None
            if hasattr(result, 'image'):
                image_obj = result.image
            elif hasattr(result, 'chart'):
                image_obj = result.chart
            elif hasattr(result, 'figure'):
                image_obj = result.figure
            
            if image_obj is not None and isinstance(image_obj, (PILImage.Image, Figure)):
                content_bytes = _convert_image_to_base64_content(image_obj)
                if content_bytes:
                    await cl.Image(
                        name=f"chart_{uuid.uuid4().hex[:8]}.png",
                        content=content_bytes,
                        display="inline"
                    ).send()
                    
                    # Optionally upload to GCS
                    gcs_url, filename = await _upload_chart_to_gcs(content_bytes, bucket)
                    image_sent = True
                    logger.info("Sent chart from object.image attribute")
            
            # Method 2: Try the value/path approach if image method didn't work
            if not image_sent:
                chart_path = None
                if hasattr(result, 'value'):
                    chart_path = str(result.value)
                elif hasattr(result, 'path'):
                    chart_path = str(result.path)
                elif hasattr(result, 'file_path'):
                    chart_path = str(result.file_path)
                
                if chart_path:
                    # Try multiple possible locations for the chart file
                    possible_paths = [
                        os.path.join(charts_dir, os.path.basename(chart_path)),
                        os.path.join(os.getcwd(), chart_path),
                        os.path.join("/tmp", chart_path),
                        chart_path
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            content_bytes = _convert_image_to_base64_content(path)
                            if content_bytes:
                                await cl.Image(
                                    name=f"chart_{uuid.uuid4().hex[:8]}.png",
                                    content=content_bytes,
                                    display="inline"
                                ).send()
                                image_sent = True
                                logger.info("Sent chart from path: %s", path)
                                break
                    
                    if not image_sent:
                        to_send_text = f"Chart generated but not accessible: {chart_path}"
                        logger.warning("Could not access chart file: %s", chart_path)
                        
                        # Debug: List files in charts directory
                        if os.path.exists(charts_dir):
                            files = os.listdir(charts_dir)
                            logger.info("Files in charts_dir: %s", files[:10])  # Show first 10 files
            
            # Method 3: Try to convert the result object itself if it's image-like
            if not image_sent:
                try:
                    # Sometimes the ChartResponse object can be converted directly
                    content_bytes = _convert_image_to_base64_content(result)
                    if content_bytes:
                        await cl.Image(
                            name=f"chart_{uuid.uuid4().hex[:8]}.png",
                            content=content_bytes,
                            display="inline"
                        ).send()
                        image_sent = True
                        logger.info("Sent chart by converting result object directly")
                except Exception as e:
                    logger.info("Could not convert result object directly: %s", e)

        elif isinstance(result, str):
            raw = result.strip()
            
            # Try to resolve as chart path first
            resolved_path = None
            if raw.startswith("exports/charts/") or raw.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Try multiple possible locations
                possible_paths = [
                    os.path.join(charts_dir, os.path.basename(raw)),
                    os.path.join(os.getcwd(), raw) if not os.path.isabs(raw) else raw,
                    os.path.join("/tmp", raw),
                    raw
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        resolved_path = path
                        break
            
            if resolved_path and os.path.exists(resolved_path):
                # Convert to content bytes for reliable display
                content_bytes = _convert_image_to_base64_content(resolved_path)
                if content_bytes:
                    # Option 1: Use content bytes directly (recommended for deployment)
                    await cl.Image(
                        name=f"chart_{uuid.uuid4().hex[:8]}.png",
                        content=content_bytes,
                        display="inline"
                    ).send()
                    
                    # Option 2: Also upload to GCS for persistence (optional)
                    gcs_url, filename = await _upload_chart_to_gcs(content_bytes, bucket)
                    if gcs_url:
                        logger.info("Chart also available at: %s", gcs_url)
                    
                    image_sent = True
                    logger.info("Sent image from path: %s", resolved_path)
            else:
                to_send_text = result or "(Empty response)"

        elif isinstance(result, PILImage.Image):
            # Handle PIL Image directly
            content_bytes = _convert_image_to_base64_content(result)
            if content_bytes:
                await cl.Image(
                    name=f"chart_{uuid.uuid4().hex[:8]}.png",
                    content=content_bytes,
                    display="inline"
                ).send()
                
                # Optionally upload to GCS
                gcs_url, filename = await _upload_chart_to_gcs(content_bytes, bucket)
                image_sent = True
                logger.info("Sent PIL image")

        elif isinstance(result, Figure):
            # Handle matplotlib Figure directly
            content_bytes = _convert_image_to_base64_content(result)
            if content_bytes:
                await cl.Image(
                    name=f"chart_{uuid.uuid4().hex[:8]}.png",
                    content=content_bytes,
                    display="inline"
                ).send()
                
                # Optionally upload to GCS
                gcs_url, filename = await _upload_chart_to_gcs(content_bytes, bucket)
                image_sent = True
                logger.info("Sent matplotlib figure")

        elif isinstance(result, dict):
            logger.info("Processing dict result with keys: %s", list(result.keys()))
            
            # Handle various dict formats
            if "image_base64" in result:
                # Direct base64 image
                try:
                    content_bytes = base64.b64decode(result["image_base64"])
                    await cl.Image(
                        name=f"chart_{uuid.uuid4().hex[:8]}.png",
                        content=content_bytes,
                        display="inline"
                    ).send()
                    image_sent = True
                    logger.info("Sent base64 image from dict")
                except Exception as e:
                    logger.error("Failed to decode base64 image: %s", e)
                    to_send_text = "Failed to decode chart image"
            
            elif "value" in result or "path" in result:
                # Handle path-based results
                chart_path = result.get("value") or result.get("path")
                if chart_path and isinstance(chart_path, str):
                    # Try to find and load the file
                    search_paths = [
                        os.path.join(charts_dir, os.path.basename(chart_path)),
                        chart_path,
                        os.path.join("/tmp", chart_path),
                        os.path.join("/tmp/exports/charts", os.path.basename(chart_path))
                    ]
                    
                    chart_found = False
                    for search_path in search_paths:
                        if os.path.exists(search_path):
                            content_bytes = _convert_image_to_base64_content(search_path)
                            if content_bytes:
                                await cl.Image(
                                    name=os.path.basename(search_path),
                                    content=content_bytes,
                                    display="inline"
                                ).send()
                                image_sent = True
                                chart_found = True
                                logger.info("Sent chart from dict path: %s", search_path)
                                break
                    
                    if not chart_found:
                        to_send_text = f"Chart generated but file not accessible: {chart_path}"
                        
                        # Debug: list available files
                        if os.path.exists(charts_dir):
                            files = os.listdir(charts_dir)
                            logger.info("Files in charts_dir (%s): %s", charts_dir, files)
            else:
                # No recognizable image data
                to_send_text = json.dumps(result, indent=2)

        else:
            to_send_text = str(result) if result is not None else "(Empty response)"

        # Send text response if we have one and no image was sent
        if to_send_text:
            await cl.Message(content=to_send_text).send()
            logger.info("Sent text response: %s", to_send_text[:100])

        # Update conversation history
        response_content = to_send_text or "[Chart generated and displayed]"
        message_history.append({"role": "assistant", "content": response_content})
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