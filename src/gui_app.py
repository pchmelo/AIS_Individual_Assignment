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
                        st.dataframe(dist_df, width='stretch')

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
                        st.image(img_path, width="stretch")
            
            # Display scale-based visualizations
            if scale_images:
                st.markdown("#### Combined Analysis by Scale")
                for img_path in scale_images:
                    if os.path.exists(img_path):
                        scale_name = os.path.basename(img_path).replace('_scale.png', '').upper()
                        st.markdown(f"**{scale_name} Scale**")
                        st.image(img_path, width="stretch")
            
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
                            st.image(img_path, width="stretch")

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
        if st.button("New Evaluation", key="new_eval", width='stretch'):
            st.session_state.mode = "new"
            st.rerun()
    
    with col2:
        if st.button("View Previous Results", key="view_results", width='stretch'):
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
            if st.button("Start Evaluation", width='stretch', type="primary"):
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
        
        # Initialize pipeline
        pipeline = DatasetEvaluationPipeline(use_api_model=st.session_state.model_choice)
        
        # Set up pipeline properties
        pipeline.current_dataset = st.session_state.dataset_name
        pipeline.target_column = st.session_state.target_column
        pipeline.user_objective = prompt
        
        # Create report directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline.report_dir = os.path.join(BASE_DIR, "reports", f"{st.session_state.dataset_name}_{timestamp}")
        pipeline.images_dir = os.path.join(pipeline.report_dir, "images")
        os.makedirs(pipeline.images_dir, exist_ok=True)
        
        st.session_state.pipeline = pipeline
        
        st.session_state.evaluation_results = {
            "dataset": st.session_state.dataset_name,
            "target_column": st.session_state.target_column,
            "user_objective": prompt,
            "report_directory": pipeline.report_dir,
            "stages": {}
        }
        
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
    
    # Display all completed stages first
    for idx, (stage_key, stage_name) in enumerate(stages):
        if idx < st.session_state.current_step and stage_key in results["stages"]:
            # Display completed stage
            display_stage_results(stage_key, results["stages"][stage_key])
            st.markdown("---")
    
    # Execute and display the current stage only
    if st.session_state.current_step < len(stages):
        stage_key, stage_name = stages[st.session_state.current_step]
        
        # Special handling for Stage 4.5 - ask for combination selection before execution
        if stage_key == "4_5_target_fairness" and stage_key not in results["stages"]:
            # Get sensitive columns from previous stage
            sensitive_cols = []
            if "3_sensitive" in results["stages"]:
                sensitive_cols = results["stages"]["3_sensitive"].get("sensitive_columns", [])
            
            if len(sensitive_cols) >= 2:
                st.markdown("### Stage 4.5: Target Fairness Analysis")
                st.info(f"Detected {len(sensitive_cols)} sensitive attributes: {', '.join(sensitive_cols)}")
                
                # Generate all possible pairs
                from itertools import combinations
                possible_pairs = list(combinations(sensitive_cols, 2))
                pair_options = [f"{a} + {b}" for a, b in possible_pairs]
                
                st.markdown("**Select which attribute combinations to analyze:**")
                st.caption("Choose the combinations you want to visualize. Only selected combinations will generate charts.")
                
                # Initialize session state for selected combinations if not exists
                if 'selected_combinations' not in st.session_state:
                    st.session_state.selected_combinations = []
                
                selected_display = st.multiselect(
                    "Attribute Combinations:",
                    options=pair_options,
                    default=st.session_state.selected_combinations,
                    help="Select which pairs of sensitive attributes to analyze together"
                )
                
                # Update session state
                st.session_state.selected_combinations = selected_display
                
                # Convert display format back to tuple format
                selected_pairs = []
                for display in selected_display:
                    parts = display.split(' + ')
                    if len(parts) == 2:
                        selected_pairs.append((parts[0], parts[1]))
                
                # Show confirmation button
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Generate Analysis", type="primary", key="gen_stage_4_5"):
                        if not selected_pairs:
                            st.warning("Please select at least one combination to analyze")
                        else:
                            # Store selected pairs in session state for execution
                            st.session_state.stage_4_5_pairs = selected_pairs
                            
                            # Execute stage with selected pairs
                            with st.spinner(f"Running {stage_name}..."):
                                try:
                                    # Preserve pipeline directories before syncing
                                    report_dir = pipeline.report_dir
                                    images_dir = pipeline.images_dir
                                    
                                    pipeline.evaluation_results = results
                                    
                                    # Restore directories
                                    pipeline.report_dir = report_dir
                                    pipeline.images_dir = images_dir
                                    
                                    # Pass selected pairs to the stage
                                    stage_result = execute_stage_with_pairs(
                                        pipeline, stage_key, st.session_state.user_prompt,
                                        results.get('dataset'), results.get('target_column'),
                                        selected_pairs
                                    )
                                    results["stages"][stage_key] = stage_result
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error in {stage_name}: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                                    return
                
                return  # Don't show continue button yet
        
        # Execute stage if not already done
        if stage_key not in results["stages"]:
            with st.spinner(f"Running {stage_name}..."):
                try:
                    # Sync pipeline's internal state with session state results
                    # This is needed because some stages depend on previous stage results
                    # But preserve the pipeline's own directories
                    report_dir = pipeline.report_dir
                    images_dir = pipeline.images_dir
                    
                    pipeline.evaluation_results = results
                    
                    # Restore directories
                    pipeline.report_dir = report_dir
                    pipeline.images_dir = images_dir
                    
                    stage_result = execute_stage(pipeline, stage_key, st.session_state.user_prompt, 
                                                 results.get('dataset'), results.get('target_column'))
                    results["stages"][stage_key] = stage_result
                except Exception as e:
                    st.error(f"Error in {stage_name}: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        # Display current stage results
        display_stage_results(stage_key, results["stages"][stage_key])
        
        # Show continue button if not the last stage
        if st.session_state.current_step < len(stages) - 1:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button(f"Continue →", key=f"continue_{stage_key}"):
                    # Clear modal state when continuing to next stage
                    st.session_state.show_modal = False
                    st.session_state.modal_image = None
                    st.session_state.modal_title = None
                    st.session_state.current_step += 1
                    st.rerun()
    
    # If all stages are complete, show completion message
    if st.session_state.current_step >= len(stages) - 1 and len(results["stages"]) == len(stages):
        st.markdown("---")
        st.success("Evaluation completed successfully!")
        
        # Generate final reports
        if "final_reports_generated" not in results:
            with st.spinner("Generating reports..."):
                try:
                    pipeline.evaluation_results = results
                    pipeline.generate_report()
                    results["final_reports_generated"] = True
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

def execute_stage_with_pairs(pipeline, stage_key, user_prompt, dataset_name, target_column, selected_pairs):
    """Execute Stage 4.5 with user-selected combination pairs"""
    if stage_key == "4_5_target_fairness":
        return pipeline._stage_4_5_target_fairness_analysis(dataset_name, target_column, selected_pairs)
    else:
        return execute_stage(pipeline, stage_key, user_prompt, dataset_name, target_column)

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
    
    # Display tool result if available
    if "tool_result" in stage_result:
        tool_result = stage_result["tool_result"]
        
        if stage_key == "0_loading":
            if tool_result.get("status") == "success":
                st.success(f"Dataset loaded: {tool_result.get('rows', 0)} rows, {len(tool_result.get('columns', []))} columns")
                with st.expander("View Columns"):
                    cols = tool_result.get('columns', [])
                    st.write(", ".join(f"`{c}`" for c in cols))
            else:
                st.error(f"Failed to load dataset: {tool_result.get('error', 'Unknown error')}")
        
        elif isinstance(tool_result, dict) and tool_result:
            with st.expander("Tool Result"):
                st.json(tool_result)
    
    # Special display for Stage 6 (Recommendations)
    if stage_key == "6_recommendations" and "recommendations" in stage_result:
        with st.expander("Recommendations", expanded=True):
            st.markdown(stage_result["recommendations"])
    
    # Display agent analysis
    if "agent_analysis" in stage_result:
        # Special formatting for Stage 3 (Sensitive Attribute Detection)
        if stage_key == "3_sensitive":
            import re
            import pandas as pd
            
            # Display column summary first in expander
            if "simplified_summary" in stage_result:
                with st.expander("Column Summary Table"):
                    # Parse the summary into a proper table
                    summary_text = stage_result["simplified_summary"]
                    lines = summary_text.strip().split('\n')
                    
                    # Find the data lines (skip header and separator lines)
                    data_lines = []
                    for line in lines:
                        if line and not line.startswith('=') and not line.startswith('COLUMN') and 'Column' not in line or 'Type' not in line:
                            if not all(c in '= \t' for c in line):
                                data_lines.append(line)
                    
                    # Parse into table
                    table_data = []
                    for line in data_lines:
                        parts = line.split()
                        if len(parts) >= 4:
                            col_name = parts[0]
                            col_type = parts[1]
                            unique = parts[2]
                            values = ' '.join(parts[3:])
                            table_data.append({
                                "Column": col_name,
                                "Type": col_type,
                                "Unique": unique,
                                "Sample Values / Top Categories": values
                            })
                    
                    if table_data:
                        df_summary = pd.DataFrame(table_data)
                        st.dataframe(df_summary, width='stretch', hide_index=True)
                    else:
                        # Fallback to text if parsing fails
                        st.text(summary_text)
            
            # Then show sensitive attributes
            analysis_text = stage_result["agent_analysis"]
            
            # Parse the sensitive attributes into a table
            pattern = r'Column:\s*([^\|]+)\s*\|\s*Reason:\s*([^\|]+)\s*\|\s*Values:\s*(.+?)(?=Column:|$)'
            matches = re.findall(pattern, analysis_text, re.DOTALL)
            
            if matches:
                st.markdown("---")
                st.markdown("**Identified Sensitive Attributes:**")
                
                # Create dataframe for table display
                table_data = []
                for col, reason, values in matches:
                    table_data.append({
                        "Column": col.strip(),
                        "Reason": reason.strip(),
                        "Values": values.strip().replace('\n', ' ')
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, width='stretch', hide_index=True)
                
                # Show count
                st.info(f"Total: {len(table_data)} sensitive attributes identified")
        
        # Special formatting for Stage 4.5 (Target Fairness Analysis)
        elif stage_key == "4_5_target_fairness":
            # Display visualizations with user selection for combinations
            tool_result = stage_result.get("tool_result", {})
            
            # Check for errors first
            if tool_result.get("status") == "error":
                st.error(f"Error generating fairness analysis: {tool_result.get('message', 'Unknown error')}")
                with st.expander("Error Details"):
                    st.json(tool_result)
            
            generated_images = tool_result.get("generated_images", [])
            
            if generated_images:
                st.markdown("---")
                st.markdown("**Visualizations:**")
                
                # Separate main images from combination images
                main_images = []
                combination_images = {}
                
                for img_path in generated_images:
                    # Check if this is a combination image (contains "_combinations" in path)
                    if '_combinations' in img_path:
                        # Extract the combination name from the path
                        path_parts = img_path.split(os.sep)
                        
                        # Find the folder with "_combinations" in it
                        combo_folder = None
                        for part in path_parts:
                            if part.endswith('_combinations'):
                                combo_folder = part.replace('_combinations', '')
                                break
                        
                        if combo_folder:
                            # Make it readable: "Age_Race" -> "Age + Race"
                            combo_display = combo_folder.replace('_', ' + ')
                            
                            if combo_display not in combination_images:
                                combination_images[combo_display] = []
                            combination_images[combo_display].append(img_path)
                    else:
                        # This is a main/simple visualization
                        main_images.append(img_path)
                
                # Display main/simple visualizations as table with clickable links
                if main_images:
                    st.markdown("#### Main Visualizations")
                    
                    # Display as table
                    for idx, img_path in enumerate(main_images):
                        if os.path.exists(img_path):
                            filename = os.path.basename(img_path)
                            display_name = filename.replace('.png', '').replace('_', ' ').title()
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{display_name}**")
                            with col2:
                                if st.button("View", key=f"view_main_{idx}"):
                                    st.session_state.modal_image = img_path
                                    st.session_state.modal_title = display_name
                                    st.session_state.show_modal = True
                                    st.rerun()
                
                # Display combination visualizations as table
                if combination_images:
                    st.markdown("---")
                    st.markdown("#### Combined Sensitive Attribute Analysis")
                    
                    # Count total combination charts
                    total_charts = sum(len(imgs) for imgs in combination_images.values())
                    st.info(f"{len(combination_images)} attribute combinations available ({total_charts} total charts)")
                    
                    # Let user select which combination pairs to view
                    selected_combos = st.multiselect(
                        "Select attribute combinations to visualize:",
                        options=sorted(combination_images.keys()),
                        help="Choose which combinations of sensitive attributes you want to analyze (e.g., Age + Race, Sex + Education)",
                        key="combo_selection"
                    )
                    
                    # Clear modal state if selection changes
                    if 'previous_combo_selection' not in st.session_state:
                        st.session_state.previous_combo_selection = []
                    
                    if st.session_state.previous_combo_selection != selected_combos:
                        st.session_state.show_modal = False
                        st.session_state.modal_image = None
                        st.session_state.modal_title = None
                        st.session_state.previous_combo_selection = selected_combos
                    
                    if selected_combos:
                        for combo_name in selected_combos:
                            st.markdown(f"##### {combo_name}")
                            combo_imgs = combination_images[combo_name]
                            
                            # Create table for this combination
                            for idx, img_path in enumerate(combo_imgs):
                                if os.path.exists(img_path):
                                    filename = os.path.basename(img_path)
                                    
                                    # Create display name
                                    if 'scale.png' in filename:
                                        display_name = filename.replace('_scale.png', '').upper() + " Scale"
                                    else:
                                        display_name = filename.replace('.png', '').replace('_', ' ').title()
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**{display_name}**")
                                    with col2:
                                        img_key = f"combo_{combo_name.replace(' + ', '_')}_{idx}"
                                        if st.button("View", key=f"view_{img_key}"):
                                            st.session_state.modal_image = img_path
                                            st.session_state.modal_title = display_name
                                            st.session_state.show_modal = True
                                            st.rerun()
                            
                            st.markdown("---")
                    else:
                        st.info("Select attribute combinations above to view their visualizations")
                else:
                    st.warning("No combination visualizations were generated (requires at least 2 sensitive attributes)")
            
            # Show agent analysis for Stage 4.5
            with st.expander("Agent Analysis", expanded=True):
                st.markdown(stage_result["agent_analysis"])
        
        # Display modal outside the stage_key check - only when explicitly requested
        if stage_key == "4_5_target_fairness" and st.session_state.get('show_modal', False):
            if hasattr(st.session_state, 'modal_image') and st.session_state.modal_image:
                @st.dialog(st.session_state.modal_title, width="large")
                def show_image_modal():
                    st.image(st.session_state.modal_image, width="stretch")
                    if st.button("Close", type="primary"):
                        st.session_state.modal_image = None
                        st.session_state.modal_title = None
                        st.session_state.show_modal = False
                        st.rerun()
                
                show_image_modal()
        
        elif stage_key != "3_sensitive" and stage_key != "4_5_target_fairness":
            # Default display for other stages (not 3 or 4.5)
            with st.expander("Agent Analysis", expanded=True):
                st.markdown(stage_result["agent_analysis"])
    
    # For stage 1 which has different structure
    if stage_key == "1_objective" and "objective" in stage_result:
        st.info(f"**Objective:** {stage_result['objective']}")
        st.write(f"**Audit Request:** {'Yes' if stage_result.get('is_audit_request') else 'No'}")
        st.write(f"**Validation:** {stage_result.get('validation', 'N/A')}")


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
