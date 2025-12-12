import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import DatasetEvaluationPipeline

# Get base directory (parent of src folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Dataset Fairness Evaluation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with dark mode support
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'selected_report' not in st.session_state:
        st.session_state.selected_report = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = 0
    if 'step_approved' not in st.session_state:
        st.session_state.step_approved = {}
    if 'pipeline_started' not in st.session_state:
        st.session_state.pipeline_started = False
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = None

def get_available_datasets():
    """Get list of available datasets from data folder"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if os.path.exists(data_dir):
        return [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return []

def get_available_reports():
    """Get list of available report folders"""
    reports_dir = os.path.join(BASE_DIR, "reports")
    if os.path.exists(reports_dir):
        return [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))]
    return []

def upload_dataset(uploaded_file):
    """Save uploaded dataset to data folder"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return uploaded_file.name

def get_dataset_columns(dataset_name):
    """Get column names from dataset"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        file_path = os.path.join(data_dir, dataset_name)
        df = pd.read_csv(file_path, nrows=1)
        return list(df.columns)
    except Exception as e:
        st.error(f"Error reading dataset: {str(e)}")
        return []

def display_stage_results(stage_name, stage_data):
    """Display results for a specific stage"""
    stage_titles = {
        "0_loading": "STAGE 0: Dataset Loading",
        "1_objective": "STAGE 1: Objective Inspection",
        "2_quality": "STAGE 2: Data Quality Analysis",
        "3_sensitive": "STAGE 3: Sensitive Attribute Detection",
        "4_imbalance": "STAGE 4: Imbalance Analysis",
        "4_5_target_fairness": "STAGE 4.5: Target Fairness Analysis",
        "5_integration": "STAGE 5: Findings Integration",
        "6_recommendations": "STAGE 6: Recommendations"
    }
    
    st.markdown(f"<div class='step-header'>{stage_titles.get(stage_name, stage_name.upper())}</div>", 
                unsafe_allow_html=True)
    
    with st.expander("View Stage Details", expanded=False):
        if isinstance(stage_data, dict):
            # Display tool information
            if "tool_used" in stage_data:
                st.markdown(f"**Tool Used:** `{stage_data['tool_used']}`")
            
            # Display tool results
            if "tool_result" in stage_data:
                st.markdown("### Tool Results")
                
                tool_result = stage_data["tool_result"]
                
                # Special handling for specific tools
                if stage_name == "2_quality":
                    display_quality_results(tool_result)
                elif stage_name == "3_sensitive":
                    display_sensitive_results(stage_data)
                elif stage_name == "4_imbalance":
                    display_imbalance_results(tool_result)
                elif stage_name == "4_5_target_fairness":
                    display_fairness_results(stage_data)
                else:
                    st.json(tool_result)
            
            # Display agent analysis
            if "agent_analysis" in stage_data:
                st.markdown("### Agent Analysis")
                st.markdown(f"<div class='info-box'>{stage_data['agent_analysis']}</div>", 
                           unsafe_allow_html=True)
            
            if "recommendations" in stage_data:
                st.markdown("### Recommendations")
                st.markdown(f"<div class='success-box'>{stage_data['recommendations']}</div>", 
                           unsafe_allow_html=True)
        else:
            st.write(stage_data)

def display_quality_results(tool_result):
    """Display quality analysis results in a structured way"""
    if tool_result.get("status") == "success":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", tool_result.get("total_rows", 0))
        with col2:
            st.metric("Missing Values", tool_result.get("total_missing_values", 0))
        with col3:
            st.metric("Missing %", f"{tool_result.get('overall_missing_percentage', 0):.2f}%")
        
        if tool_result.get("details"):
            st.markdown("#### Issues by Column")
            
            # Create DataFrame for better display
            issues_data = []
            for detail in tool_result["details"]:
                issues_data.append({
                    "Column": detail["column"],
                    "Data Type": detail["data_type"],
                    "Missing Count": detail["missing_count"],
                    "Missing %": f"{detail['missing_percentage']:.2f}%",
                    "Issues": detail.get("detected_issues", "")
                })
            
            if issues_data:
                df_issues = pd.DataFrame(issues_data)
                st.dataframe(df_issues, use_container_width=True)

def display_sensitive_results(stage_data):
    """Display sensitive attribute detection results"""
    tool_result = stage_data.get("tool_result", {})
    sensitive_cols = stage_data.get("sensitive_columns", [])
    
    if sensitive_cols:
        st.markdown(f"**Identified Sensitive Columns:** {', '.join(sensitive_cols)}")
        st.markdown("---")
    
    if "simplified_summary" in stage_data:
        st.markdown("#### Column Summary")
        st.text(stage_data["simplified_summary"])

def display_imbalance_results(tool_result):
    """Display imbalance analysis results"""
    if tool_result.get("status") == "success":
        st.metric("Imbalanced Columns", tool_result.get("imbalanced_columns", 0))
        
        if tool_result.get("details"):
            st.markdown("#### Imbalanced Columns Details")
            
            for detail in tool_result["details"]:
                with st.expander(f"**{detail['column']}**"):
                    st.write(f"**Dominant Value:** {detail['dominant_value']}")
                    st.write(f"**Dominant Percentage:** {detail['dominant_percentage']:.2f}%")
                    
                    if "distribution" in detail:
                        dist_df = pd.DataFrame([
                            {"Value": k, "Percentage": v} 
                            for k, v in detail["distribution"].items()
                        ])
                        st.dataframe(dist_df, use_container_width=True)

def display_fairness_results(stage_data):
    """Display target fairness analysis results with images"""
    tool_result = stage_data.get("tool_result", {})
    
    if tool_result.get("status") == "success":
        st.markdown(f"**Target Column:** {tool_result.get('target_column')}")
        st.markdown(f"**Sensitive Columns:** {', '.join(tool_result.get('sensitive_columns', []))}")
        
        # Display generated images
        generated_images = tool_result.get("generated_images", [])
        
        if generated_images:
            st.markdown("### Visualizations")
            
            # Group images by type
            scale_images = [img for img in generated_images if 'scale.png' in img]
            individual_images = [img for img in generated_images if 'individual_combinations' in img]
            other_images = [img for img in generated_images if img not in scale_images and img not in individual_images]
            
            # Display main visualizations
            if other_images:
                st.markdown("#### Main Visualizations")
                for img_path in other_images:
                    if os.path.exists(img_path):
                        st.image(img_path, use_column_width=True)
            
            # Display scale-based visualizations
            if scale_images:
                st.markdown("#### Combined Analysis by Scale")
                for img_path in scale_images:
                    if os.path.exists(img_path):
                        scale_name = os.path.basename(img_path).replace('_scale.png', '').upper()
                        st.markdown(f"**{scale_name} Scale**")
                        st.image(img_path, use_column_width=True)
            
            # Display individual combinations (selectable)
            if individual_images:
                st.markdown("#### Individual Combinations")
                st.info(f"{len(individual_images)} individual combination graphs available")
                
                # Let user select which combinations to view
                selected_combos = st.multiselect(
                    "Select combinations to view:",
                    options=individual_images[:20],  # Limit to first 20 for UI performance
                    format_func=lambda x: os.path.basename(x).replace('.png', '').replace('_', ' ')
                )
                
                if selected_combos:
                    for img_path in selected_combos:
                        if os.path.exists(img_path):
                            st.image(img_path, use_column_width=True)

def main_page():
    """Main landing page"""
    st.markdown("<div class='main-header'>Dataset Quality & Fairness Evaluation System</div>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h3>Welcome</h3>
    <p>This tool helps you evaluate datasets for data quality issues and fairness concerns.</p>
    <p><strong>Choose an option below to get started:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("New Evaluation", key="new_eval", use_container_width=True):
            st.session_state.mode = "new"
            st.rerun()
    
    with col2:
        if st.button("View Previous Results", key="view_results", use_container_width=True):
            st.session_state.mode = "view"
            st.rerun()

def new_evaluation_page():
    """Page for creating new evaluation"""
    st.markdown("<div class='main-header'>New Evaluation</div>", unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Configuration")
        
        if st.button("← Back to Main"):
            st.session_state.mode = None
            st.session_state.current_step = 0
            st.session_state.pipeline = None
            st.rerun()
        
        st.markdown("---")
        
        # Dataset selection
        st.markdown("#### Dataset")
        datasets = get_available_datasets()
        
        uploaded_file = st.file_uploader("Upload new dataset", type=['csv'])
        if uploaded_file:
            dataset_name = upload_dataset(uploaded_file)
            st.success(f"Uploaded: {dataset_name}")
            datasets = get_available_datasets()
        
        selected_dataset = st.selectbox("Select dataset", datasets)
        st.session_state.dataset_name = selected_dataset
        
        # Model selection
        st.markdown("#### Model Selection")
        model_options = {
            0: "IBM Granite (Local)",
            1: "Grok (API)",
            2: "Google Gemini (API)"
        }
        st.session_state.model_choice = st.radio(
            "Choose model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        # Target column selection
        st.markdown("#### Target Column (Optional)")
        use_target = st.checkbox("Specify target column for fairness analysis")
        
        if use_target and selected_dataset:
            columns = get_dataset_columns(selected_dataset)
            if columns:
                st.session_state.target_column = st.selectbox("Select target column:", columns)
            else:
                st.warning("Could not read dataset columns")
                st.session_state.target_column = None
        else:
            st.session_state.target_column = None
        
        st.markdown("---")
        
        # Start evaluation button
        if selected_dataset:
            if st.button("Start Evaluation", use_container_width=True, type="primary"):
                initialize_pipeline()

    # Main content area
    if st.session_state.pipeline_started:
        display_pipeline_stepwise()
    elif st.session_state.evaluation_results:
        display_pipeline_results()
    else:
        st.markdown("""
        <div class='info-box'>
        <h3>Configure your evaluation in the sidebar</h3>
        <p>1. Select or upload a dataset</p>
        <p>2. Choose an AI model</p>
        <p>3. Optionally specify a target column</p>
        <p>4. Click "Start Evaluation" to begin</p>
        </div>
        """, unsafe_allow_html=True)

def initialize_pipeline():
    """Initialize the pipeline and prepare for step-by-step execution"""
    try:
        # Create user prompt
        prompt = f"Evaluate the dataset '{st.session_state.dataset_name}' for data quality and fairness issues."
        if st.session_state.target_column:
            prompt += f" Target: {st.session_state.target_column}."
        prompt += " Provide a detailed report highlighting any problems found and suggestions for improvement."
        
        st.session_state.user_prompt = prompt
        st.session_state.pipeline_started = True
        st.session_state.current_step = 0
        st.session_state.step_approved = {}
        st.session_state.evaluation_results = {
            "dataset": st.session_state.dataset_name,
            "target_column": st.session_state.target_column,
            "user_objective": prompt,
            "stages": {}
        }
        
        # Initialize pipeline
        pipeline = DatasetEvaluationPipeline(use_api_model=st.session_state.model_choice)
        st.session_state.pipeline = pipeline
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        st.exception(e)

def display_pipeline_stepwise():
    """Display and execute pipeline step by step"""
    results = st.session_state.evaluation_results
    pipeline = st.session_state.pipeline
    
    if not pipeline or not results:
        return
    
    # Display metadata
    with st.expander("Evaluation Metadata", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Dataset:** {results.get('dataset')}")
        with col2:
            st.markdown(f"**Target:** {results.get('target_column', 'None')}")
        with col3:
            model_names = {0: "IBM Granite", 1: "Grok", 2: "Google Gemini"}
            st.markdown(f"**Model:** {model_names.get(st.session_state.model_choice, 'Unknown')}")
    
    st.markdown("---")
    
    # Define stages
    stages = [
        ("0_loading", "Dataset Loading"),
        ("1_objective", "Objective Inspection"),
        ("2_quality", "Data Quality Analysis"),
        ("3_sensitive", "Sensitive Attribute Detection"),
        ("4_imbalance", "Imbalance Analysis"),
        ("4_5_target_fairness", "Target Fairness Analysis") if st.session_state.target_column else None,
        ("5_integration", "Findings Integration"),
        ("6_recommendations", "Recommendations")
    ]
    stages = [s for s in stages if s is not None]
    
    # Execute and display stages up to current step
    for idx, (stage_key, stage_name) in enumerate(stages):
        if idx <= st.session_state.current_step:
            # Execute stage if not already done
            if stage_key not in results["stages"]:
                with st.spinner(f"Running {stage_name}..."):
                    try:
                        stage_result = execute_stage(pipeline, stage_key, st.session_state.user_prompt, 
                                                     results.get('dataset'), results.get('target_column'))
                        results["stages"][stage_key] = stage_result
                    except Exception as e:
                        st.error(f"Error in {stage_name}: {str(e)}")
                        return
            
            # Display stage results
            display_stage_results(stage_key, results["stages"][stage_key])
            
            # Show continue button if this is the current step and not the last
            if idx == st.session_state.current_step and idx < len(stages) - 1:
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"Continue →", key=f"continue_{stage_key}"):
                        st.session_state.current_step += 1
                        st.rerun()
                # Don't show next stages until user clicks continue
                break
            
            st.markdown("---")
    
    # If all stages are complete, show completion message
    if st.session_state.current_step >= len(stages) - 1:
        st.success("Evaluation completed successfully!")
        
        # Generate final reports
        if "report_directory" not in results:
            with st.spinner("Generating reports..."):
                try:
                    pipeline.evaluation_results = results
                    pipeline.generate_report()
                    results["report_directory"] = pipeline.report_dir
                except Exception as e:
                    st.error(f"Error generating reports: {str(e)}")
        
        if "report_directory" in results:
            st.markdown(f"**Report Directory:** `{results['report_directory']}`")

def execute_stage(pipeline, stage_key, user_prompt, dataset_name, target_column):
    """Execute a single pipeline stage"""
    if stage_key == "0_loading":
        return pipeline._stage_0_load_dataset(dataset_name)
    elif stage_key == "1_objective":
        return pipeline._stage_1_objective_inspection(user_prompt)
    elif stage_key == "2_quality":
        return pipeline._stage_2_data_quality(dataset_name)
    elif stage_key == "3_sensitive":
        return pipeline._stage_3_sensitive_detection(dataset_name)
    elif stage_key == "4_imbalance":
        return pipeline._stage_4_imbalance_analysis(dataset_name)
    elif stage_key == "4_5_target_fairness":
        return pipeline._stage_4_5_target_fairness_analysis(dataset_name, target_column)
    elif stage_key == "5_integration":
        return pipeline._stage_5_integrate_findings()
    elif stage_key == "6_recommendations":
        return pipeline._stage_6_recommendations()
    else:
        return {"status": "error", "message": f"Unknown stage: {stage_key}"}

def display_stage_results(stage_key, stage_result):
    """Display the results of a specific stage"""
    # Extract stage name from key
    stage_names = {
        "0_loading": "Stage 0: Dataset Loading",
        "1_objective": "Stage 1: Objective Inspection",
        "2_quality": "Stage 2: Data Quality Analysis",
        "3_sensitive": "Stage 3: Sensitive Attribute Detection",
        "4_imbalance": "Stage 4: Imbalance Analysis",
        "4_5_target_fairness": "Stage 4.5: Target Fairness Analysis",
        "5_integration": "Stage 5: Findings Integration",
        "6_recommendations": "Stage 6: Recommendations"
    }
    
    st.markdown(f"### {stage_names.get(stage_key, stage_key)}")
    
    # Display based on stage type
    if stage_key == "0_loading":
        if stage_result.get("status") == "success":
            st.success(f"Dataset loaded: {stage_result.get('rows', 0)} rows, {stage_result.get('columns', 0)} columns")
        else:
            st.error(f"Failed to load dataset: {stage_result.get('error', 'Unknown error')}")
    
    elif stage_key in ["1_objective", "5_integration", "6_recommendations"]:
        if "summary" in stage_result:
            st.info(stage_result["summary"])
        if "details" in stage_result:
            with st.expander("Details"):
                st.write(stage_result["details"])
    
    elif stage_key == "2_quality":
        if "issues" in stage_result:
            if stage_result["issues"]:
                st.warning(f"Found {len(stage_result['issues'])} quality issues")
                for issue in stage_result["issues"]:
                    st.markdown(f"- {issue}")
            else:
                st.success("No quality issues found")
    
    elif stage_key == "3_sensitive":
        if "sensitive_attributes" in stage_result:
            attrs = stage_result["sensitive_attributes"]
            if attrs:
                st.warning(f"Detected {len(attrs)} sensitive attributes: {', '.join(attrs)}")
            else:
                st.success("No sensitive attributes detected")
    
    elif stage_key in ["4_imbalance", "4_5_target_fairness"]:
        if "summary" in stage_result:
            st.info(stage_result["summary"])
        if "charts" in stage_result:
            st.write(f"Generated {len(stage_result['charts'])} visualizations")

def run_pipeline_evaluation():
    """Run the evaluation pipeline"""
    with st.spinner("Initializing pipeline..."):
        try:
            # Create user prompt
            prompt = f"Evaluate the dataset '{st.session_state.dataset_name}' for data quality and fairness issues."
            if st.session_state.target_column:
                prompt += f" Target: {st.session_state.target_column}."
            prompt += " Provide a detailed report highlighting any problems found and suggestions for improvement."
            
            # Initialize pipeline
            pipeline = DatasetEvaluationPipeline(use_api_model=st.session_state.model_choice)
            st.session_state.pipeline = pipeline
            
            # Run evaluation
            results = pipeline.evaluate_dataset(prompt)
            st.session_state.evaluation_results = results
            st.session_state.current_step = 0
            
            st.success("Evaluation completed successfully")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            st.exception(e)

def display_pipeline_results():
    """Display pipeline results step by step"""
    results = st.session_state.evaluation_results
    
    if not results:
        return
    
    # Display metadata
    with st.expander("Evaluation Metadata", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Dataset:** {results.get('dataset')}")
        with col2:
            st.markdown(f"**Target:** {results.get('target_column', 'None')}")
        with col3:
            model_names = {0: "IBM Granite", 1: "Grok", 2: "Google Gemini"}
            st.markdown(f"**Model:** {model_names.get(st.session_state.model_choice, 'Unknown')}")
        
        if "report_directory" in results:
            st.markdown(f"**Report Directory:** `{results['report_directory']}`")
    
    st.markdown("---")
    
    # Display stages
    stages = results.get("stages", {})
    stage_keys = sorted(stages.keys())
    
    for idx, stage_key in enumerate(stage_keys):
        stage_data = stages[stage_key]
        
        # Display stage
        display_stage_results(stage_key, stage_data)
        
        # Add continue button for each step (except last)
        if idx < len(stage_keys) - 1:
            if stage_key not in st.session_state.step_approved:
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"Continue →", key=f"continue_{stage_key}"):
                        st.session_state.step_approved[stage_key] = True
                        st.rerun()
                
                # Don't show next stages until approved
                break
        
        st.markdown("---")

def view_results_page():
    """Page for viewing previous results"""
    st.markdown("<div class='main-header'>Previous Results</div>", unsafe_allow_html=True)
    
    if st.button("← Back to Main"):
        st.session_state.mode = None
        st.session_state.selected_report = None
        st.rerun()
    
    reports = get_available_reports()
    
    if not reports:
        st.warning("No previous reports found.")
        return
    
    selected_report = st.selectbox("Select a report to view:", reports)
    
    if selected_report:
        report_dir = os.path.join(BASE_DIR, "reports", selected_report)
        
        # Display report files
        report_file = os.path.join(report_dir, "evaluation_report.txt")
        summary_file = os.path.join(report_dir, "agent_summary.txt")
        
        tab1, tab2, tab3 = st.tabs(["Full Report", "Agent Summary", "Visualizations"])
        
        with tab1:
            if os.path.exists(report_file):
                with open(report_file, 'r', encoding='utf-8') as f:
                    st.text(f.read())
            else:
                st.warning("Report file not found")
        
        with tab2:
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    st.text(f.read())
            else:
                st.warning("Summary file not found")
        
        with tab3:
            images_dir = os.path.join(report_dir, "images")
            if os.path.exists(images_dir):
                # List all image files
                image_files = []
                for root, dirs, files in os.walk(images_dir):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append(os.path.join(root, file))
                
                if image_files:
                    st.markdown(f"**Found {len(image_files)} visualizations**")
                    
                    # Group by folder
                    folders = {}
                    for img in image_files:
                        folder = os.path.dirname(img).replace(images_dir, '').strip(os.sep)
                        if folder not in folders:
                            folders[folder] = []
                        folders[folder].append(img)
                    
                    for folder, images in folders.items():
                        folder_name = folder if folder else "Main Images"
                        with st.expander(f"{folder_name}"):
                            for img_path in images:
                                st.image(img_path, caption=os.path.basename(img_path))
                else:
                    st.info("No images found in this report")
            else:
                st.info("No images directory found")

def main():
    """Main application entry point"""
    init_session_state()
    
    if st.session_state.mode is None:
        main_page()
    elif st.session_state.mode == "new":
        new_evaluation_page()
    elif st.session_state.mode == "view":
        view_results_page()

if __name__ == "__main__":
    main()
