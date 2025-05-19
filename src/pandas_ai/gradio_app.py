import gradio as gr
import json
import logging
import glob
from omegaconf import DictConfig
import logging
import pandas as pd
import os
import hydra

logger = logging.getLogger(__name__)


def preprocess_data(file_path: str, output_path: str) -> pd.DataFrame:
    """Preprocess the data by standardizing column names, specific column values"""
    logger.info(f"Processing file: {file_path}")
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        return None
    
    # Standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    # Process specific columns
    if 'shop_name' in df.columns:
        df['shop_name'] = df['shop_name'].str.lower().str.replace(' ', '_')
    return df


def combine_and_sort_files(cfg, files=None):
    """Preprocess all files in the raw directory, combine them, sort by
    Creation Date, and save as a single CSV"""
    raw_dir = os.path.join(cfg.data_dir, 'raw')
    processed_dir = os.path.join(cfg.data_dir, 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get all CSV and Excel files in the raw directory
    all_files = []
    if files:
        for file in files:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ['.csv', '.xlsx', '.xls']:
                continue
            # Save the uploaded file to raw directory
            filename = os.path.basename(file.name)
            raw_path = os.path.join(raw_dir, filename)
            
            # Copy the file to raw directory
            import shutil
            shutil.copy(file.name, raw_path)
            all_files.append(raw_path)
    else:
        for extension in ['*.csv', '*.xlsx', '*.xls']:
            all_files.extend(glob.glob(os.path.join(raw_dir, extension)))
    
    if not all_files:
        logger.warning("No CSV or Excel files found in the raw directory.")
        return None
    
    logger.info(f"Found {len(all_files)} files to process.")
    
    # Process each file and collect dataframes
    dfs = []
    for file_path in all_files:
        try:
            df = preprocess_data(cfg, file_path)
            if df is not None:
                dfs.append(df)
                logger.info(f"Successfully processed {filename}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    if not dfs:
        logger.warning("No valid dataframes to combine.")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if "Creation Date" or similar column exists
    creation_date_col = None
    for col in combined_df.columns:
        if 'creation' in col.lower() and 'date' in col.lower():
            creation_date_col = col
            break
    
    if not creation_date_col:
        logger.warning("No 'Creation Date' column found. Combined data will not be sorted.")
    else:
        # Convert to datetime if necessary
        if not pd.api.types.is_datetime64_dtype(combined_df[creation_date_col]):
            try:
                combined_df[creation_date_col] = pd.to_datetime(combined_df[creation_date_col])
                logger.info(f"Converted {creation_date_col} to datetime format.")
            except Exception as e:
                logger.error(f"Error converting {creation_date_col} to datetime: {str(e)}")
                logger.warning("Combined data will not be sorted.")
                creation_date_col = None
        
        if creation_date_col:
            # Sort by creation date
            combined_df = combined_df.sort_values(by=creation_date_col)
            logger.info(f"Sorted combined data by {creation_date_col}.")
    
    # Save combined DataFrame
    combined_path = os.path.join(processed_dir, 'combined_data.csv')
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"Combined data saved to {combined_path}")
    
    return combined_path


def save_metadata(
    cfg: DictConfig,
    file_path: str,
    column_descriptions: List[str],
    dataset_description: str
) -> str:
    """
    Save the metadata for the dataset
    
    Args:
        cfg: Hydra configuration
        file_path: Path to the dataset file
        column_descriptions: Dictionary of column descriptions
        dataset_description: General description of the dataset
        
    Returns:
        Success message
    """
    df = pd.read_csv(file_path)
    
    # Create column metadata in the required format
    columns = []
    for col_name, description in column_descriptions.items():
        # Skip empty descriptions
        if not description:
            continue
            
        # Determine column type
        if pd.api.types.is_numeric_dtype(df[col_name]):
            col_type = "float" if pd.api.types.is_float_dtype(df[col_name]) else "integer"
        elif pd.api.types.is_datetime64_dtype(df[col_name]):
            col_type = "datetime"
        else:
            col_type = "string"
            
        columns.append({
            "name": col_name,
            "type": col_type,
            "description": description
        })
    
    # Save metadata to a JSON file for Chainlit to use
    metadata = {
        "description": dataset_description,
        "columns": columns,
        "file_path": file_path
    }
    
    metadata_path = os.path.join(cfg.data_dir, 'processed', 'metadata.json')
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    return f"Metadata saved successfully! Your data is ready for querying."


def create_data_upload_interface(cfg):
    """
    Create the Gradio interface for uploading and processing data
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Gradio app
    """
    def process_files(files):
        """Process uploaded files and return the path and columns"""
        if not files or len(files) == 0:
            return None, None, "Please upload at least one file."
        
        try:
            # Process, combine and sort all files
            combined_path = combine_and_sort_files(cfg, files)
            
            if not combined_path or not os.path.exists(combined_path):
                return None, None, "No valid files were processed."
            
            # Load the combined CSV to get columns
            df = pd.read_csv(combined_path)
            columns = list(df.columns)
            
            return combined_path, columns, f"Files processed and combined successfully! Combined {len(files)} files."
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            return None, None, f"Error processing files: {str(e)}"

    def create_description_interface(file_path, columns):
        """
        Create an interface for entering column descriptions
        """
        if columns is None:
            return gr.update(visible=False)
        
        with gr.Blocks() as block:
            gr.Markdown("## Enter Column Descriptions")
            
            # Dataset description
            dataset_description = gr.Textbox(
                label="Dataset Description", 
                placeholder="Provide a general description of this dataset",
                lines=3
            )
            
            # Create input fields for each column
            column_inputs = {}
            for col in columns:
                column_inputs[col] = gr.Textbox(
                    label=f"Description for '{col}'",
                    placeholder=f"Describe what the '{col}' column represents"
                )
            
            # Submit button
            submit_btn = gr.Button("Save and Continue")
            
            # Output message
            output_msg = gr.Textbox(label="Status")
            
            # Handle submission
            def on_submit():
                descriptions = {col: column_inputs[col].value for col in columns}
                return save_metadata(cfg, file_path, descriptions, dataset_description.value)
            
            submit_btn.click(
                fn=on_submit,
                inputs=None,
                outputs=[output_msg]
            )
        
        return block
    
    def upload_files(files):
        """
        Process the uploaded files and create an interface for entering column descriptions
        """
        if not files or len(files) == 0:
            return gr.update(visible=False), "Please upload at least one file."
        
        file_path, columns, message = process_files(files)
        
        if file_path is None:
            return gr.update(visible=False), message
        
        # Create column description interface
        description_interface = create_description_interface(file_path, columns)
        
        return gr.update(visible=True, value=description_interface), message
    
    # Create Gradio interface
    with gr.Blocks(title="Data Upload and Processing") as app:
        gr.Markdown("# Data Upload and Processing")
        gr.Markdown("Upload your Excel or CSV files. They will be combined, sorted by Creation Date, and prepared for querying.")
        
        with gr.Row():
            file_input = gr.Files(label="Upload Datasets (Excel or CSV)")
        
        with gr.Row():
            upload_btn = gr.Button("Process Files")
        
        with gr.Row():
            status_output = gr.Textbox(label="Status")
        
        with gr.Row():
            description_container = gr.Container(visible=False)
        
        upload_btn.click(
            fn=upload_files,
            inputs=[file_input],
            outputs=[description_container, status_output]
        )
        
        gr.Markdown("""
        ## Instructions
        1. Upload your dataset files (Excel or CSV)
        2. Click "Process Files" to preprocess the data
           - Files will be combined and sorted by "Creation Date" if available
        3. Enter descriptions for each column (this helps the AI understand your data)
        4. Click "Save and Continue"
        5. After saving, you can close this window and launch the Chat interface
        """)
    
    return app

