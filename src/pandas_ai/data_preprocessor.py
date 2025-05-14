import os
import pandas as pd



def preprocess_column_names(cfg):
    raw_dir = os.path.join(cfg.data_dir, 'raw')
    processed_dir = os.path.join(cfg.data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):
        if filename.endswith('.csv'):
            raw_path = os.path.join(raw_dir, filename)
            df = pd.read_csv(raw_path)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            processed_path = os.path.join(processed_dir, filename)
            df.to_csv(processed_path, index=False)