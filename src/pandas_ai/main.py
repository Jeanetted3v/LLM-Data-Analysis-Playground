import os
import logging
import hydra
from omegaconf import DictConfig
import gradio as gr
import subprocess
import sys


logger = logging.getLogger(__name__)

def check_data_exists(cfg):
    """Check if processed data exists"""
    metadata_path = os.path.join(cfg.data_dir, 'processed', 'metadata.json')
    return os.path.exists(metadata_path)

def launch_data_upload(cfg):
    """Launch the data upload interface"""
    from src.apps.gradio_app import create_data_upload_interface
    app = create_data_upload_interface(cfg)
    app.launch()
    return "Data upload interface launched in a new tab. Please close this window and continue there."

def launch_chat(cfg):
    """Launch the chat interface"""
    if not check_data_exists(cfg):
        return "No data found. Please upload and process data first."
    
    logger.info("Starting Chainlit application...")
    
    # Launch Chainlit with the script
    try:
        chainlit_script = os.path.join(os.path.dirname(__file__), "apps", "chainlit_app.py")
        subprocess.run([sys.executable, "-m", "chainlit", "run", chainlit_script])
        return "Chat session ended. You can start another chat session if needed."
    except Exception as e:
        logger.error(f"Error launching Chainlit: {str(e)}")
        return f"Error launching chat interface: {str(e)}"

def create_main_interface(cfg):
    """
    Create the main selection interface
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Gradio app
    """
    data_exists = check_data_exists(cfg)
    
    with gr.Blocks(title="Natural Language Dataset Query System") as app:
        gr.Markdown("# Natural Language Dataset Query System")
        
        if data_exists:
            gr.Markdown("## Your data is ready for querying!")
            gr.Markdown("You can upload new data or chat with your existing data.")
        else:
            gr.Markdown("## Welcome to the Natural Language Dataset Query System")
            gr.Markdown("You need to upload and process your data before you can query it.")
        
        with gr.Row():
            upload_btn = gr.Button("Upload & Process New Data")
            chat_btn = gr.Button("Chat with Data", interactive=data_exists)
        
        output = gr.Textbox(label="Status", interactive=False)
        
        upload_btn.click(fn=lambda: launch_data_upload(cfg), inputs=None, outputs=output)
        chat_btn.click(fn=lambda: launch_chat(cfg), inputs=None, outputs=output)
        
        if not data_exists:
            gr.Markdown("""
            ## Getting Started
            1. Click "Upload & Process New Data" to begin
            2. Upload your Excel or CSV files
            3. Provide descriptions for your data columns
            4. Return to this screen and click "Chat with Data"
            """)
        else:
            gr.Markdown("""
            ## Ready to Chat!
            Click "Chat with Data" to start asking questions about your data in natural language.
            
            Examples of questions you can ask:
            - "What's the average value in column X?"
            - "Show me a summary of data from the last month"
            - "Which products have the highest sales?"
            - "Plot the trend of X over time"
            """)
    
    return app

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main entry point of the application
    
    Args:
        cfg: Hydra configuration
    """
    # Create data directories
    os.makedirs(os.path.join(cfg.data_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(cfg.data_dir, 'processed'), exist_ok=True)
    
    logger.info("Starting application...")
    
    # Create and launch the main interface
    app = create_main_interface(cfg)
    app.launch()

if __name__ == "__main__":
    main()