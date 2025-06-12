import streamlit as st

def render_schema_display(schema: dict):
    """Display the current database schema in a user-friendly format."""
    st.sidebar.header("📊 Database Schema")
    
    # Create an expander for schema details
    with st.sidebar.expander("View Database Schema", expanded=False):
        if not schema:
            st.warning("No schema information available.")
            return
            
        for table, columns in schema.items():
            st.markdown(f"**Table: `{table}`**")
            st.markdown("Columns:")
            for col in columns:
                st.markdown(f"- `{col}`")
            st.markdown("---") 