# app/main.py

import streamlit as st
from pathlib import Path
from utils.config import UPLOAD_DIR, METADATA_DIR, get_gemini_api_key, save_gemini_api_key_to_config
from utils.db_utils import load_excel_to_duckdb, extract_schema_and_samples, save_metadata, load_metadata, load_schema_metadata
from utils.langchain_utils import get_langchain_chain
from utils.memory import load_memory, save_memory
from utils.sql_utils import extract_column_names, validate_sql_query
from components.sidebar import render_sidebar
from components.data_overview import render_data_overview
from components.chat_interface import render_chat_interface, display_last_response
from components.sql_execution import render_sql_execution
from components.memory_panel import render_memory_panel
from styles.custom_css import get_custom_css
import pandas as pd  # Import pandas for displaying results and handling DataFrames
import duckdb_engine  # Import to allow warning filtering (optional)
import warnings  # Import warnings (optional)
import re  # Import regex module for SQL parsing
import json  # Import json for saving/loading memory
from json import JSONEncoder  # Import JSONEncoder for custom serialization
from pandas import Timestamp  # Import Timestamp to check and handle
from sqlalchemy import text
import traceback  # Import traceback module


# Define the path for the memory file
MEMORY_FILE = Path("./chat_memory.json")

# Custom JSON Encoder to handle non-serializable types like pandas Timestamp


class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Timestamp):
            return str(obj)  # Convert Timestamp to string
        # Let the base class default method raise the TypeError for other types
        return JSONEncoder.default(self, obj)


# Filter out the duckdb-engine reflection warning (optional)
warnings.filterwarnings("ignore", category=duckdb_engine.DuckDBEngineWarning)

st.set_page_config(page_title="Data API Chat", layout="wide")
st.title("📊 Data API Chat")

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize or Load Session State
if "chat_history" not in st.session_state or "column_usage_counts" not in st.session_state:
    st.session_state["chat_history"], st.session_state["column_usage_counts"] = load_memory()
    st.sidebar.success("Memory loaded.")

# Initialize API request counter
if "api_request_count" not in st.session_state:
    st.session_state["api_request_count"] = 0
if "current_question" not in st.session_state:
    st.session_state["current_question"] = None

# Render sidebar and get file/API key info
uploaded_file, selected_file, api_key_input = render_sidebar()

# Load or process file
active_file = uploaded_file.name if uploaded_file else selected_file
meta_path = METADATA_DIR / f"{active_file}.pkl" if active_file else None
schema_meta_path = METADATA_DIR / f"{active_file}_schema.json" if active_file else None

conn, schema, samples, distributions, metadata_embeddings = None, None, None, None, None
schema_metadata = None

if active_file:
    # Get the full file path - either from upload directory or selected file
    if uploaded_file:
        file_path = UPLOAD_DIR / uploaded_file.name
    else:
        file_path = UPLOAD_DIR / selected_file

    # Load schema metadata if available
    if schema_meta_path and schema_meta_path.exists():
        schema_metadata = load_schema_metadata(schema_meta_path)
        if schema_metadata:
            st.sidebar.success("Loaded schema metadata.")

    # Attempt to load metadata first
    loaded_conn, loaded_schema, loaded_samples, loaded_distributions, loaded_metadata_embeddings = load_metadata(meta_path)

    if loaded_conn:
        conn = loaded_conn
        schema = loaded_schema
        samples = loaded_samples
        distributions = loaded_distributions
        metadata_embeddings = loaded_metadata_embeddings

        if conn:
            st.sidebar.success(f"Loaded data for: {active_file}")
    else:
        st.sidebar.info(f"Processing: {active_file}")
        try:
            conn = load_excel_to_duckdb(file_path)
            if conn:
                schema, samples, distributions, metadata_embeddings = extract_schema_and_samples(conn)

            if conn and schema and samples and distributions and metadata_embeddings:
                save_metadata(file_path, schema, samples, distributions, metadata_embeddings, meta_path)
                st.sidebar.success(f"Processed and cached: {active_file}")
            elif not conn:
                st.sidebar.error(f"Failed to load data from {active_file}")
            else:
                st.sidebar.error(f"Failed to extract complete metadata from {active_file}")
                conn, schema, samples, distributions, metadata_embeddings = None, None, None, None, None

        except Exception as e:
            st.sidebar.error(f"Error processing file {active_file}: {e}")
            conn, schema, samples, distributions, metadata_embeddings = None, None, None, None, None

# Main content area
if active_file and conn and api_key_input:
    # Data Overview
    render_data_overview(samples)

    # Chat Interface
    render_chat_interface()

    # Process user input if available
    user_input_to_process = st.session_state.get("current_user_input", "")

    if st.session_state.get("process_query", False) and user_input_to_process and active_file and conn and api_key_input:
        # Check if this is a new question
        if user_input_to_process != st.session_state["current_question"]:
            st.session_state["current_question"] = user_input_to_process
            st.session_state["api_request_count"] = 0

        # Check API request limit
        if st.session_state["api_request_count"] >= 7:
            st.error("Maximum number of API requests (7) reached for this question. Please try rephrasing your question or start a new one.")
            st.session_state["process_query"] = False
            st.session_state["current_user_input"] = ""
        else:
            st.session_state["process_query"] = False
            st.session_state["current_user_input"] = ""
            st.session_state["api_request_count"] += 1

            try:
                st.info("Invoking LangChain chain to generate SQL...")
                langchain_chain = get_langchain_chain(
                    conn,
                    api_key=api_key_input,
                    schema=schema,
                    samples=samples,
                    distributions=distributions,
                    metadata_embeddings=metadata_embeddings,
                    chat_history=st.session_state.get("chat_history", []),
                    column_usage_counts=st.session_state.get("column_usage_counts", {}))

                chain_raw_output = langchain_chain.invoke({"question": user_input_to_process})

                sql_query = None
                if isinstance(chain_raw_output, str):
                    sql_query = chain_raw_output
                elif isinstance(chain_raw_output, dict) and 'output' in chain_raw_output and isinstance(chain_raw_output['output'], str):
                    sql_query = chain_raw_output['output']

                if sql_query and isinstance(sql_query, str):
                    extracted_cols = extract_column_names(sql_query, schema)
                    response_dict = {
                        "sql_query": sql_query,
                        "columns_used": extracted_cols,
                        "feedback": None,
                        "corrected_query": None,
                        "corrected_columns": None,
                        "column_mapping": {col: 1 for col in extracted_cols}
                    }

                    try:
                        query_lower = sql_query.strip().lower()
                        if query_lower.startswith("select") or query_lower.startswith("with"):
                            result_df = pd.read_sql(sql_query, conn)
                            response_dict["result"] = result_df
                        else:
                            result_proxy = conn.execute(text(sql_query))
                            rows_affected = result_proxy.rowcount if hasattr(result_proxy, 'rowcount') else 'N/A'
                            response_dict["result"] = f"Query executed successfully. Rows affected: {rows_affected}"

                        st.session_state["last_response"] = response_dict
                        if "chat_history" not in st.session_state:
                            st.session_state["chat_history"] = []
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": response_dict})
                        save_memory(
                            st.session_state["chat_history"],
                            st.session_state["column_usage_counts"])

                    except Exception as e:
                        error_traceback = traceback.format_exc()
                        full_error_message = f"An error occurred during query execution: {e}\n\nTraceback:\n{error_traceback}"
                        st.error(full_error_message)
                        st.code(sql_query, language="sql")

                        st.session_state["last_response"] = {
                            "error": full_error_message,
                            "columns_used": extracted_cols,
                            "feedback": "incorrect",
                            "corrected_query": None,
                            "corrected_columns": None,
                            "column_mapping": {col: 1 for col in extracted_cols}}
                        if "chat_history" not in st.session_state:
                            st.session_state["chat_history"] = []
                        st.session_state["chat_history"].append({
                            "role": "assistant",
                            "content": {
                                "sql_query": sql_query,
                                "result": full_error_message,
                                "columns_used": extracted_cols,
                                "feedback": "incorrect",
                                "corrected_query": None,
                                "corrected_columns": None,
                                "column_mapping": {col: 1 for col in extracted_cols}
                            }
                        })
                        save_memory(
                            st.session_state["chat_history"],
                            st.session_state["column_usage_counts"])

            except Exception as e:
                error_traceback = traceback.format_exc()
                full_error_message = f"An error occurred during query execution: {e}\n\nTraceback:\n{error_traceback}"
                st.error(full_error_message)
                st.code(sql_query, language="sql")

                st.session_state["last_response"] = {
                    "error": full_error_message,
                    "columns_used": [],
                    "feedback": "incorrect",
                    "corrected_query": None,
                    "corrected_columns": None,
                    "column_mapping": {}
                }
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": {
                        "sql_query": "Error",
                        "result": full_error_message,
                        "columns_used": [],
                        "feedback": "incorrect",
                        "corrected_query": None,
                        "corrected_columns": None,
                        "column_mapping": {}
                    }
                })
                save_memory(
                    st.session_state["chat_history"],
                    st.session_state["column_usage_counts"])

    # Display last response
    display_last_response(st.session_state.get("last_response"))

    # Direct SQL Execution
    render_sql_execution(conn, schema)
