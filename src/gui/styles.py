import streamlit as st


def inject_css():
    """Inject all custom CSS styles into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
    /* Main header styling */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        text-align: left;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }
    
    /* Stage header styling */
    .step-header {
        background-color: #3498db;
        color: #ffffff;
        padding: 0.875rem 1.25rem;
        border-radius: 4px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.125rem;
        font-weight: 600;
        border-left: 4px solid #2980b9;
    }
    
    /* Info boxes - works in both light and dark mode */
    .info-box {
        background-color: rgba(52, 152, 219, 0.1);
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-left: 4px solid #3498db;
        padding: 1.25rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .info-box h3 {
        margin-top: 0;
        font-size: 1.125rem;
        font-weight: 600;
        color: #3498db;
    }
    
    .info-box p, .info-box strong {
        opacity: 0.95;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: rgba(243, 156, 18, 0.1);
        border: 1px solid rgba(243, 156, 18, 0.3);
        border-left: 4px solid #f39c12;
        padding: 1.25rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Success boxes */
    .success-box {
        background-color: rgba(39, 174, 96, 0.1);
        border: 1px solid rgba(39, 174, 96, 0.3);
        border-left: 4px solid #27ae60;
        padding: 1.25rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #3498db !important;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.625rem 1.25rem;
        border: none;
        transition: background-color 0.2s ease;
        font-size: 0.9375rem;
    }
    
    .stButton>button:hover {
        background-color: #2980b9 !important;
        border-color: #2980b9 !important;
    }
    
    .stButton>button:active {
        background-color: #21618c !important;
    }
    
    /* Primary button styling */
    .stButton>button[kind="primary"] {
        background-color: #27ae60 !important;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #229954 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 0.875rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 4px;
        font-weight: 500;
        background-color: rgba(52, 152, 219, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        background-color: rgba(128, 128, 128, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: #ffffff !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(52, 152, 219, 0.3);
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Select box and input styling */
    .stSelectbox > div > div,
    .stTextInput > div > div {
        border-radius: 4px;
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-weight: 500;
    }
    
    /* Success/Error/Warning/Info message styling */
    .stAlert {
        border-radius: 4px;
    }

    /* ---- Chatbot input area ---- */
    [data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 0.75rem;
        background-color: rgba(52, 152, 219, 0.03);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
"""
