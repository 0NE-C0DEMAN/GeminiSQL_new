import streamlit as st
from utils.memory import save_memory
import pandas as pd

def render_chat_interface():
    """Renders the chat interface with user input form."""
    st.header("💬 Input Question")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about your data...", key="chat_input_text")
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            st.session_state["process_query"] = True
            st.session_state["current_user_input"] = user_input
            
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            save_memory(st.session_state["chat_history"], st.session_state["column_usage_counts"])
            st.rerun()

def display_last_response(last_response):
    """Displays the last query response and feedback section."""
    if not last_response:
        return

    # Display the user's last question
    last_question = next(
        (msg["content"] for msg in reversed(st.session_state.get("chat_history", [])) if msg["role"] == "user"),
        ""
    )
    if last_question:
        st.markdown(f"**Your Last Question:** {last_question}")

    if "error" in last_response:
        st.error(last_response["error"])
        if last_response.get("columns_used"):
            st.markdown(f"**Extracted Columns (on error):** {', '.join(last_response['columns_used'])}")
    elif isinstance(last_response, dict) and "sql_query" in last_response and "result" in last_response:
        # Display generated SQL query
        with st.container(border=True):
            st.markdown("**Generated SQL Query:**")
            st.code(last_response["sql_query"], language="sql")

        # Display extracted columns
        if last_response.get("columns_used"):
            st.markdown(f"**Extracted Columns:** {', '.join(last_response['columns_used'])}")

        # Display query result
        with st.container(border=True):
            st.markdown("**Query Result:**")
            if isinstance(last_response["result"], pd.DataFrame):
                st.dataframe(last_response["result"], use_container_width=True)
            else:
                result_to_display = last_response["result"]
                if isinstance(result_to_display, dict) and result_to_display.get("data") is not None:
                    try:
                        result_df = pd.DataFrame(result_to_display["data"], columns=result_to_display["columns"])
                        st.dataframe(result_df, use_container_width=True)
                    except Exception:
                        st.code(str(result_to_display))
                else:
                    st.code(str(result_to_display))

        # Feedback Section
        st.markdown("---")
        st.markdown("**Provide Feedback (Optional):**")

        col1, col2 = st.columns(2)
        with col1:
            correct_button = st.button("👍 Correct", key="feedback_correct")
        with col2:
            incorrect_button = st.button("👎 Incorrect", key="feedback_incorrect")

        corrected_query_input = st.text_area(
            "Provide a corrected SQL query:",
            key="feedback_corrected_query",
            height=100
        )
        submit_feedback_button = st.button("Submit Feedback", key="feedback_submit")

        # Process feedback
        if correct_button:
            process_feedback("correct")
        elif incorrect_button:
            process_feedback("incorrect")
        elif submit_feedback_button and corrected_query_input:
            process_feedback("corrected", corrected_query_input)

def process_feedback(feedback_type, corrected_query=None):
    """Processes user feedback and updates the chat history."""
    st.session_state["last_response"]["feedback"] = feedback_type
    
    if feedback_type == "corrected":
        st.session_state["last_response"]["corrected_query"] = corrected_query
        # Extract and update column usage for corrected query
        if corrected_query:
            from utils.sql_utils import extract_column_names
            from utils.db_utils import extract_schema_and_samples
            corrected_cols = extract_column_names(corrected_query, st.session_state.get("schema", {}))
            # Increment usage for corrected columns
            for col in corrected_cols:
                st.session_state["column_usage_counts"][col] = st.session_state["column_usage_counts"].get(col, 0) + 3  # Give even more weight to corrected columns
            # Store the corrected columns in the response
            st.session_state["last_response"]["corrected_columns"] = corrected_cols
            
            # Decrement usage for incorrect columns
            if "columns_used" in st.session_state["last_response"]:
                incorrect_cols = set(st.session_state["last_response"]["columns_used"]) - set(corrected_cols)
                for col in incorrect_cols:
                    st.session_state["column_usage_counts"][col] = max(0, st.session_state["column_usage_counts"].get(col, 0) - 2)
                    # Add a negative mapping to remember this column was incorrect
                    st.session_state["last_response"]["column_mapping"][col] = -2
            
            # Add positive mapping for corrected columns
            for col in corrected_cols:
                st.session_state["last_response"]["column_mapping"][col] = 3
            
            # Extract any specific values used in the correction (e.g., 'TRUST')
            if "ticket_compliance_status = 'TRUST'" in corrected_query:
                st.session_state["last_response"]["value_mapping"] = {
                    "ticket_compliance_status": "TRUST"
                }
            
        st.toast("Feedback received: Corrected query submitted.")
    elif feedback_type == "incorrect":
        # Decrement usage for incorrect columns
        if "columns_used" in st.session_state["last_response"]:
            for col in st.session_state["last_response"]["columns_used"]:
                if col in st.session_state["column_usage_counts"]:
                    st.session_state["column_usage_counts"][col] = max(0, st.session_state["column_usage_counts"][col] - 2)
                    # Add a negative mapping to remember this column was incorrect
                    st.session_state["last_response"]["column_mapping"][col] = -2
        st.toast("Feedback received: Query marked as incorrect.")
    else:  # correct
        # Increment usage for correct columns
        if "columns_used" in st.session_state["last_response"]:
            for col in st.session_state["last_response"]["columns_used"]:
                st.session_state["column_usage_counts"][col] = st.session_state["column_usage_counts"].get(col, 0) + 2
                # Add a positive mapping to remember this column was correct
                st.session_state["last_response"]["column_mapping"][col] = 2
        st.toast("Feedback received: Query marked as correct.")

    # Update chat history
    for msg in reversed(st.session_state.get("chat_history", [])):
        if msg["role"] == "assistant" and "sql_query" in msg["content"]:
            msg["content"]["feedback"] = feedback_type
            if feedback_type == "corrected":
                msg["content"]["corrected_query"] = corrected_query
                msg["content"]["corrected_columns"] = corrected_cols if corrected_query else None
                # Copy over any value mappings
                if "value_mapping" in st.session_state["last_response"]:
                    msg["content"]["value_mapping"] = st.session_state["last_response"]["value_mapping"]
            elif "corrected_query" in msg["content"]:
                del msg["content"]["corrected_query"]
                if "corrected_columns" in msg["content"]:
                    del msg["content"]["corrected_columns"]
                if "value_mapping" in msg["content"]:
                    del msg["content"]["value_mapping"]
            break

    save_memory(st.session_state["chat_history"], st.session_state["column_usage_counts"])
    st.rerun() 