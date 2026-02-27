"""
Main (landing) page.
"""

import streamlit as st


def main_page():
    """Render the landing page with navigation buttons."""
    st.markdown(
        "<div class='main-header'>Dataset Quality &amp; Fairness Evaluation System</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='info-box'>
        <h3>Welcome</h3>
        <p>This tool helps you evaluate datasets for data quality issues and fairness concerns.</p>
        <p><strong>Choose an option below to get started:</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("New Evaluation", key="new_eval", width="stretch"):
            st.session_state.mode = "new"
            st.rerun()

    with col2:
        if st.button("View Previous Results", key="view_results", width="stretch"):
            st.session_state.mode = "view"
            st.rerun()
