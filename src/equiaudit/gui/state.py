import streamlit as st

from equiaudit.gui.config import get_default_model_name


def init_session_state():
    """Initialize all session state variables with sensible defaults."""
    _defaults = {
        # Navigation
        "mode": None,

        # Pipeline
        "pipeline": None,
        "pipeline_started": False,

        # Dataset / model
        "dataset_name": None,
        "target_column": None,
        "model_choice": get_default_model_name(),
        "user_prompt": None,

        # Stage-level overrides
        "confirmed_sensitive_columns": None,
        "ml_config": {"enabled": False},

        "evaluation_results": None,
        "selected_report": None,

        # Chatbot UI state
        "chat_messages": [],       # list of {"role": "user"|"assistant"|"system", "content": str}
        "nav_action": "forward",   # current action selector value
    }

    for key, default in _defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
