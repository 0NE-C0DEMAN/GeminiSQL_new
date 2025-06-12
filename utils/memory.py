import json
from pathlib import Path
from pandas import Timestamp
from json import JSONEncoder
import streamlit as st
import pandas as pd

# Define the path for the memory file
MEMORY_FILE = Path("./chat_memory.json")

class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Timestamp):
            return str(obj)
        return JSONEncoder.default(self, obj)

def save_memory(chat_history, column_usage_counts):
    """Saves chat history and column usage counts to a JSON file."""
    memory_data = {
        "chat_history": [],
        "column_usage_counts": column_usage_counts
    }
    
    # Prepare chat history for saving (convert DataFrames to serializable format)
    for entry in chat_history:
        serializable_entry = entry.copy()
        if serializable_entry["role"] == "assistant" and isinstance(serializable_entry["content"], dict):
            content = serializable_entry["content"]
            if isinstance(content.get("result"), pd.DataFrame):
                content["result"] = content["result"].to_dict(orient="split")
        memory_data["chat_history"].append(serializable_entry)

    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory_data, f, indent=4, cls=DateTimeEncoder)
    except Exception as e:
        st.error(f"Error saving memory: {e}")

def load_memory():
    """Loads chat history and column usage counts from a JSON file."""
    if not MEMORY_FILE.exists():
        return [], {}

    try:
        with open(MEMORY_FILE, "r") as f:
            memory_data = json.load(f)
        
        chat_history = memory_data.get("chat_history", [])
        column_usage_counts = memory_data.get("column_usage_counts", {})

        # Convert saved DataFrame representations back to DataFrames
        for entry in chat_history:
            if entry["role"] == "assistant" and isinstance(entry["content"], dict):
                content = entry["content"]
                if isinstance(content.get("result"), dict) and content["result"].get("data") is not None:
                    try:
                        result_dict = content["result"]
                        content["result"] = pd.DataFrame(result_dict["data"], columns=result_dict["columns"])
                    except Exception as e:
                        st.warning(f"Could not load DataFrame from memory entry: {e}")
                        content["result"] = "Error loading DataFrame: " + str(e)

        return chat_history, column_usage_counts
    except Exception as e:
        st.error(f"Error loading memory: {e}")
        return [], {} 