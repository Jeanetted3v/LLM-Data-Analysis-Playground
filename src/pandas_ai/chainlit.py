import chainlit as cl
import os
import pandas as pd
import pandasai as pai
from pandasai_litellm import LiteLLM
import json
import logging
import hydra
from omegaconf import DictConfig

from src.utils.settings import SETTINGS

logger = logging.getLogger(__name__)

class ChainlitApp:
    """
    Handler class for the Chainlit application
    """
    def __init__(self, config):
        """
        Initialize with configuration
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.setup_llm()
        
    def setup_llm(self):
        """Set up the LLM with configuration"""
        os.environ["OPENAI_API_KEY"] = SETTINGS.OPENAI_API_KEY
        
        # Initialize LLM
        self.llm = LiteLLM(model=self.config.llm_model)
        pai.config.set({
           "llm": self.llm,
           "temperature": self.config.temperature,
           "seed": self.config.seed,
           "save_logs": self.config.save_logs,
           "verbose": self.config.verbose,
           "max_retries": self.config.max_retries
        })
    
    async def on_chat_start(self):
        """Handler for chat start event"""
        # Check if metadata file exists
        metadata_path = os.path.join(self.config.data_dir, 'processed', 'metadata.json')
        
        if not os.path.exists(metadata_path):
            await cl.Message(
                content=f"No dataset metadata found. Please upload a dataset using the Data Upload interface first."
            ).send()
            return
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Load the dataframe
        file_path = metadata["file_path"]
        if not os.path.exists(file_path):
            await cl.Message(
                content=f"Dataset file not found. Please upload a dataset using the Data Upload interface again."
            ).send()
            return
        
        df = pd.read_csv(file_path)
        
        # Display dataframe info
        await cl.Message(content=f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.").send()
        
        # Create a SmartDataframe with metadata
        columns = metadata["columns"]
        description = metadata["description"]
        
        # Initialize PandasAI dataframe
        smart_df = pai.create(
            path="user/dataset",
            df=df,
            description=description,
            columns=columns
        )
        
        # Store the SmartDataframe in the user session
        cl.user_session.set("smart_df", smart_df)
        
        # Set initial message history
        cl.user_session.set(
            "message_history",
            [{"role": "system", "content": "You are a helpful assistant that answers questions about data."}],
        )
        
        # Display dataset info
        columns_info = "\n".join([
            f"- {col['name']}: {col['description']}" for col in columns
        ])
        
        await cl.Message(
            content=f"""## Dataset loaded successfully!
            
**Description**: {description}

**Columns**:
{columns_info}

You can now ask questions about your data.
            """
        ).send()
    
    async def on_message(self, message):
        """Handler for incoming messages"""
        # Retrieve message history and dataframe
        message_history = cl.user_session.get("message_history")
        smart_df = cl.user_session.get("smart_df")
        
        if smart_df is None:
            await cl.Message(
                content="No dataset found. Please upload a dataset using the Data Upload interface first."
            ).send()
            return
        
        # Add user message to history
        message_history.append({"role": "user", "content": message.content})
        
        # Show thinking indicator
        thinking_msg = cl.Message(content="Thinking...")
        await thinking_msg.send()
        
        try:
            # Create full prompt with history
            question = message.content
            full_prompt = "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in message_history]
            )
            full_prompt += f"\nUser: {question}"
            
            # Get response from PandasAI
            response = smart_df.chat(full_prompt)
            
            # Update the thinking message with the actual response
            await thinking_msg.update(content=response)
            
            # Update message history
            message_history.append({"role": "assistant", "content": response})
            cl.user_session.set("message_history", message_history)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await thinking_msg.update(content=f"Error: {str(e)}")
            await cl.Message(
                content="I encountered an error while processing your question. Please try again or rephrase your question."
            ).send()


@hydra.main(config_path="../../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    try:
        app = ChainlitApp(cfg)
        
        # Register Chainlit handlers
        @cl.on_chat_start
        async def on_chat_start_wrapper():
            await app.on_chat_start()
        
        @cl.on_message
        async def on_message_wrapper(message):
            await app.on_message(message)

    except Exception as e:
        print(f"Error initializing app: {str(e)}")
        
        # Fallback handlers in case of initialization error
        @cl.on_chat_start
        async def error_on_start():
            await cl.Message(
                content="Error initializing the application. Please check your configuration."
            ).send()


if __name__ == "__main__":
    main()