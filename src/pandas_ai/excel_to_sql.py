"""To run:
python -m src.pandas_ai.excel_to_sql
"""
import pandas as pd
import sqlite3
import os

def excel_to_sqlite(excel_file, db_file, sheet_name=None, table_name=None):
    """Convert Excel file to SQLite database.

    Arg:
        excel_file : str
            Path to the Excel file.
        db_file : str
            Path to the SQLite database file (will be created if it doesn't exist).
        sheet_name : str or list, optional
            Name of the sheet(s) to read. If None, all sheets will be imported.
        table_name : str or list, optional
            Name of the table(s) to create. If None, sheet names will be used.
    """
    conn = sqlite3.connect(db_file)
    
    if sheet_name is None:
        xl = pd.ExcelFile(excel_file)
        sheet_names = xl.sheet_names
    else:
        sheet_names = [sheet_name] if isinstance(sheet_name, str) else sheet_name
    
    if table_name is None:
        table_names = [name.lower().replace(' ', '_') for name in sheet_names]
    else:
        table_names = [table_name] if isinstance(table_name, str) else table_name
        if len(table_names) != len(sheet_names):
            raise ValueError("Length of table_name must match length of sheet_name")
    
    for i, (sheet, table) in enumerate(zip(sheet_names, table_names)):
        df = pd.read_excel(excel_file, sheet_name=sheet)
        df.columns = [col.lower().replace(' ', '_').replace('.', '_').replace('-', '_') 
                      for col in df.columns]

        df.to_sql(table, conn, if_exists='replace', index=False)
        print(f"Imported sheet '{sheet}' to table '{table}' with {len(df)} rows")
    conn.close()
    print(f"Successfully converted {excel_file} to {db_file}")


if __name__ == "__main__":
    excel_to_sqlite("data/raw/test_data_name_change.xlsx", "test_data_name_change.db")