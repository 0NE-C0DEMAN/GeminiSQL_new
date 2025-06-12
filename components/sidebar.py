import streamlit as st
from utils.config import UPLOAD_DIR, get_gemini_api_key, save_gemini_api_key_to_config
from utils.memory import save_memory
from components.memory_panel import render_memory_panel
import shutil
from pathlib import Path

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the upload directory."""
    if uploaded_file is not None:
        # Create upload directory if it doesn't exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def render_sidebar():
    """Renders the sidebar with file upload, selection, and memory panel."""
    st.sidebar.header("📂 Files")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file is not None:
        # Save the uploaded file to the upload directory
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            st.sidebar.success(f"File saved: {uploaded_file.name}")
    
    # File selection section
    file_list = [f for f in UPLOAD_DIR.glob("*.xls*")]
    file_names = [f.name for f in file_list]
    selected_file = st.sidebar.selectbox("Select file", file_names) if file_names else None

    # Clear Chat Button
    if st.sidebar.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.session_state["column_usage_counts"] = {}
        save_memory(st.session_state["chat_history"], st.session_state["column_usage_counts"])
        st.rerun()

    # Authentication
    st.sidebar.header("🔑 Authentication")
    current_api_key = get_gemini_api_key()
    api_key_input = st.sidebar.text_input(
        "Enter Gemini API Key",
        value=current_api_key if current_api_key else "",
        type="password"
    )

    if api_key_input and api_key_input != current_api_key:
        if save_gemini_api_key_to_config(api_key_input):
            st.sidebar.success("API Key saved!")
            st.session_state["gemini_api_key"] = api_key_input
            st.rerun()
        else:
            st.sidebar.error("Failed to save API Key.")

    # Render Memory Panel
    render_memory_panel()

    return uploaded_file, selected_file, api_key_input 