import streamlit as st

def render_memory_panel():
    """Renders the memory panel in the sidebar."""
    st.sidebar.header("🧠 Memory")
    with st.sidebar.expander("View Session Memory"):
        # Display recent questions
        st.markdown("**Recent Questions:**")
        if st.session_state.get("chat_history"):
            # Get user questions from chat history, excluding assistant responses
            user_questions = [msg["content"] for msg in reversed(st.session_state["chat_history"]) if msg["role"] == "user"]
            # Limit to the last 10 questions for brevity
            if user_questions:
                for i, question in enumerate(user_questions[:10]):
                    # Display in reverse chronological order (most recent first)
                    st.markdown(f"{i + 1}. {question}")
            else:
                st.info("No questions in history yet.")

        st.markdown("---") # Separator

        # Display column usage counts, sorted by frequency
        st.markdown("**Column Usage:**")
        if st.session_state.get("column_usage_counts"):
            # Sort columns by count in descending order
            sorted_columns = sorted(st.session_state["column_usage_counts"].items(), key=lambda item: item[1], reverse=True)
            # Limit to the top 20 columns for display in the sidebar
            if sorted_columns:
                for col, count in sorted_columns[:20]:
                    st.markdown(f"- `{col}`: {count}")
                if len(sorted_columns) > 20:
                    st.markdown("...") # Indicate there are more columns
            else:
                 st.info("No column usage data yet.")
        else:
            st.info("No column usage data yet.")

        # Future: Add more insights here (e.g., frequently used filters, joins, etc.) 