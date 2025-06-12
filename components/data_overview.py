import streamlit as st

def render_data_overview(samples):
    """Renders the data overview section with sample data in tabs."""
    st.header("📊 Data Overview")

    with st.expander("🔎 Sample Data", expanded=True):
        if samples:
            sheet_names = list(samples.keys())
            tabs = st.tabs(sheet_names)

            for i, sheet_name in enumerate(sheet_names):
                with tabs[i]:
                    st.markdown(f"**{sheet_name}**")
                    st.dataframe(samples[sheet_name], height=250, use_container_width=True)
        else:
            st.info("Sample data not available.") 