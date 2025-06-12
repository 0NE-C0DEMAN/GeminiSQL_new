# app/utils/db_utils.py
import pandas as pd
import duckdb
import joblib
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sentence_transformers import SentenceTransformer # Import SentenceTransformer
import numpy as np # Import numpy
import torch
import streamlit as st # Import streamlit here for the workaround
import json
torch.classes.__path__ = [] # Workaround for Streamlit/Torch compatibility issue

# Define the directory for DuckDB files
DUCKDB_DIR = Path(__file__).parent.parent / "duckdb_dbs"
DUCKDB_DIR.mkdir(exist_ok=True)

# Initialize the sentence transformer model (load once)
# Using a common model, can be changed later if needed
try:
    # Attempt to load the model from the default cache directory first
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading sentence transformer model: {e}")
    print("Attempting to download the model...")
    # If loading fails, try downloading explicitly (this requires an internet connection)
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer model downloaded and loaded successfully.")
    except Exception as download_error:
        print(f"Error downloading and loading sentence transformer model: {download_error}")
        sentence_model = None # Set to None if model cannot be loaded

def get_duckdb_path_from_excel_path(excel_filepath):
    # Generate a .duckdb filename from the Excel filename
    return DUCKDB_DIR / f"{Path(excel_filepath).stem}.duckdb"

def load_excel_to_duckdb(excel_filepath):
    duckdb_filepath = get_duckdb_path_from_excel_path(excel_filepath)

    # Use native duckdb to load data from excel into the .duckdb file
    # Connect to the .duckdb file. If it exists, open it. If not, create it.
    conn = duckdb.connect(database=str(duckdb_filepath))

    xls = pd.ExcelFile(excel_filepath)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        table_name = sheet.lower().replace(" ", "_")
        # Use native duckdb to register DataFrame and create table
        conn.register(f"df_{table_name}", df)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_{table_name}")
        conn.unregister(f"df_{table_name}")

    conn.close()

    # Create SQLAlchemy engine pointing to the .duckdb file
    engine = create_engine(f'duckdb:///{duckdb_filepath}')

    # Return the SQLAlchemy engine
    return engine

def save_schema_metadata(schema: dict, metadata_path: Path) -> None:
    """
    Save schema metadata including table relationships and column information to a JSON file.
    """
    metadata = {
        "tables": {},
        "relationships": [],
        "column_info": {}
    }
    
    # Store table and column information
    for table, columns in schema.items():
        metadata["tables"][table] = {
            "columns": columns,
            "primary_keys": [],  # Could be enhanced to detect primary keys
            "foreign_keys": []   # Could be enhanced to detect foreign keys
        }
        
        # Store column information
        for column in columns:
            metadata["column_info"][f"{table}.{column}"] = {
                "table": table,
                "column": column,
                "full_name": f"{table}.{column}"
            }
    
    # Save to JSON file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_schema_metadata(metadata_path: Path) -> dict:
    """
    Load schema metadata from JSON file.
    """
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def extract_schema_and_samples(engine):
    # This function receives a SQLAlchemy Engine
    schema = {}
    samples = {}
    distributions = {} # Initialize distributions dictionary
    metadata_embeddings = {} # Initialize embeddings dictionary

    if sentence_model is None:
        print("Warning: Sentence transformer model not loaded. Skipping embedding generation.")
        return schema, samples, distributions, metadata_embeddings # Return empty embeddings if model not loaded

    with engine.connect() as conn:
        inspector = inspect(conn)
        tables = inspector.get_table_names()

        for table in tables:
            # Use SQLAlchemy inspector for columns
            columns_info = inspector.get_columns(table)
            column_names = [col['name'] for col in columns_info]
            schema[table] = column_names

            # Store schema metadata in JSON
            metadata_path = DUCKDB_DIR / f"{table}_schema.json"
            save_schema_metadata(schema, metadata_path)

            # Use SQLAlchemy connection to fetch samples as a DataFrame
            # This requires pandas to be installed for SQLAlchemy's fetchall() to_frame()
            result = conn.execute(text(f"SELECT * FROM \"{table}\" LIMIT 10"))
            sample_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            samples[table] = sample_df

            # Calculate categorical distributions from samples
            table_distributions = {} # Dictionary for this table's distributions
            # Iterate through columns of the sample DataFrame to find potential categorical columns
            for col_name, col_data in sample_df.items():
                # Heuristic: Consider columns with object dtype or a limited number of unique values as categorical
                # Using sample data for efficiency, might not cover all unique values in large datasets
                if col_data.dtype == 'object' or col_data.nunique() < 20: # Threshold can be adjusted
                    # Calculate value counts, drop NaN, get top N values
                    value_counts = col_data.value_counts().head(10).to_dict() # Limit to top 10 values
                    if value_counts: # Only add if there are non-zero counts
                         table_distributions[col_name] = value_counts

            if table_distributions:
                distributions[table] = table_distributions

            # Generate embeddings for table and column names
            table_description = f"Table: {table}"
            column_descriptions = [f"Column: {col_name} in table {table}" for col_name in column_names]
            all_descriptions = [table_description] + column_descriptions

            # Generate embeddings for the descriptions
            # Ensure inputs are strings
            string_descriptions = [str(d) for d in all_descriptions]
            embeddings = sentence_model.encode(string_descriptions, convert_to_numpy=True)

            # Store embeddings with their corresponding descriptions
            metadata_embeddings[table] = {
                "description": table_description,
                "embedding": embeddings[0] # Embedding for the table description
            }
            for i, col_name in enumerate(column_names):
                 metadata_embeddings[f"{table}.{col_name}"] = {
                     "description": column_descriptions[i], # Description for the column
                     "embedding": embeddings[i+1] # Embedding for the column description
                 }

    # Return schema, samples, distributions, AND metadata_embeddings
    return schema, samples, distributions, metadata_embeddings

def save_metadata(filepath, schema, samples, distributions, metadata_embeddings, meta_path):
    meta = {
        "filepath": str(filepath),
        "schema": schema,
        "samples": samples,
        "distributions": distributions, # Include distributions in metadata
        "metadata_embeddings": metadata_embeddings # Include embeddings in metadata
    }
    with open(meta_path, "wb") as f:
        joblib.dump(meta, f)

def load_metadata(meta_path):
    if Path(meta_path).exists():
        with open(meta_path, "rb") as f:
            meta = joblib.load(f)
        # Re-establish the SQLAlchemy engine using the .duckdb file path
        excel_filepath = Path(meta["filepath"])
        duckdb_filepath = get_duckdb_path_from_excel_path(excel_filepath)

        if duckdb_filepath.exists():
             # Recreate the SQLAlchemy engine connected to the .duckdb file
            engine = create_engine(f'duckdb:///{duckdb_filepath}')
            # Although schema, samples, distributions, and embeddings are in metadata,
            # we need the engine to be returned and the re-extracted info
            # Need to re-extract schema/samples/distributions/embeddings as they are tied to the *current* state of the DuckDB file
            # if the Excel file is re-uploaded and the DB file updated.
            try:
                # Now extract schema, samples, distributions, AND embeddings from the engine
                current_schema, current_samples, current_distributions, current_metadata_embeddings = extract_schema_and_samples(engine)
                # In load_metadata, we return the engine and the re-extracted info
                return engine, current_schema, current_samples, current_distributions, current_metadata_embeddings
            except Exception as e:
                 print(f"Error re-extracting schema/samples/distributions/embeddings from {duckdb_filepath}: {e}")
                 # Return None for all if extraction fails
                 return None, None, None, None, None
        else:
            # Handle case where original DuckDB file is missing
            print(f"Warning: DuckDB file not found at {duckdb_filepath}")
            return None, None, None, None, None
    # Return None for all if metadata file doesn't exist
    return None, None, None, None, None