import streamlit as st

from gui.styles import inject_css
from gui.state import init_session_state
from gui.screens.main_page import main_page
from gui.screens.new_evaluation import new_evaluation_page
from gui.screens.view_results import view_results_page


def main():
    """Launch the Streamlit application."""
    st.set_page_config(
        page_title="Dataset Fairness Evaluation",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    init_session_state()

    if st.session_state.mode is None:
        main_page()
    elif st.session_state.mode == "new":
        new_evaluation_page()
    elif st.session_state.mode == "view":
        view_results_page()


if __name__ == "__main__":
    main()
