import streamlit as st
import pandas as pd
from sqlalchemy import text
import traceback
from utils.sql_utils import extract_column_names

def render_sql_execution(conn, schema):
    """Renders the direct SQL execution section."""
    st.header("🔧 Direct SQL Execution")

    with st.expander("Run Custom SQL Query", expanded=True):
        sql_query_input = st.text_area("Enter your SQL query here:", key="sql_query_text_area", height=150)
        execute_button = st.button("Execute SQL Query")

        # Initialize session state for custom query result
        if "custom_sql_result" not in st.session_state:
            st.session_state["custom_sql_result"] = None
        if "custom_sql_error" not in st.session_state:
            st.session_state["custom_sql_error"] = None

        if execute_button and sql_query_input and conn:
            execute_custom_sql(conn, sql_query_input, schema)

        display_sql_result()

def execute_custom_sql(conn, sql_query_input, schema):
    """Executes a custom SQL query and stores the result in session state."""
    try:
        query_lower = sql_query_input.strip().lower()
        st.info(f"Debug: Lowercased and stripped query starts with: '{query_lower[:20]}...'")

        if query_lower.startswith("select") or query_lower.startswith("with"):
            result_df = pd.read_sql(sql_query_input, conn)
            st.session_state["custom_sql_result"] = result_df
            st.session_state["custom_sql_error"] = None

            # Extract and display columns for direct queries
            extracted_cols = extract_column_names(sql_query_input, schema)
            if extracted_cols:
                st.markdown(f"**Extracted Columns:** {', '.join(extracted_cols)}")
        else:
            result_proxy = conn.execute(text(sql_query_input))
            rows_affected = result_proxy.rowcount if hasattr(result_proxy, 'rowcount') else 'N/A'
            st.session_state["custom_sql_result"] = f"Query executed successfully. Rows affected: {rows_affected}"
            st.session_state["custom_sql_error"] = None

            # Extract and display columns for direct queries
            extracted_cols = extract_column_names(sql_query_input, schema)
            if extracted_cols:
                st.markdown(f"**Extracted Columns:** {', '.join(extracted_cols)}")

    except Exception as e:
        error_traceback = traceback.format_exc()
        full_error_message = f"Error executing query: {e}\n\nTraceback:\n{error_traceback}"
        st.session_state["custom_sql_result"] = None
        st.session_state["custom_sql_error"] = full_error_message

    st.rerun()

def display_sql_result():
    """Displays the result or error of the custom SQL query."""
    if st.session_state["custom_sql_error"]:
        st.error(st.session_state["custom_sql_error"])
    elif st.session_state["custom_sql_result"] is not None:
        st.markdown("**Execution Result:**")
        if isinstance(st.session_state["custom_sql_result"], pd.DataFrame):
            st.dataframe(st.session_state["custom_sql_result"], use_container_width=True)
        else:
            st.code(str(st.session_state["custom_sql_result"]))
    elif st.session_state.get("execute_button") and not st.session_state.get("sql_query_input"):
        st.warning("Please enter a SQL query to execute.")
    elif st.session_state.get("execute_button") and not st.session_state.get("conn"):
        st.error("Database connection not available. Please upload/select a file first.") 