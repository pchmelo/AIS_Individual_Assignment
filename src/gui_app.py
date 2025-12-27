import streamlit as st
import os
import pandas as pd
import sys
from pipeline import DatasetEvaluationPipeline

sys.path.insert(0, os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Dataset Fairness Evaluation",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if os.path.exists(data_dir):
        return [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return []

def get_available_reports():
    reports_dir = os.path.join(BASE_DIR, "reports")
    if os.path.exists(reports_dir):
        return [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))]
    return []

def upload_dataset(uploaded_file):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return uploaded_file.name

def get_dataset_columns(dataset_name):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        file_path = os.path.join(data_dir, dataset_name)
        df = pd.read_csv(file_path, nrows=1)
        return list(df.columns)
    except Exception as e:
        st.error(f"Error reading dataset: {str(e)}")
        return []

def display_stage_results(stage_name, stage_data):
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
    tool_result = stage_data.get("tool_result", {})
    sensitive_cols = stage_data.get("sensitive_columns", [])
    
    if sensitive_cols:
        st.markdown(f"**Identified Sensitive Columns:** {', '.join(sensitive_cols)}")
        st.markdown("---")
    
    if "simplified_summary" in stage_data:
        st.markdown("#### Column Summary")
        st.text(stage_data["simplified_summary"])

def display_imbalance_results(tool_result):
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
        ("5_recommendations", "Recommendations"),
        ("6_bias_mitigation", "Bias Mitigation") if st.session_state.target_column else None
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
                st.info(f"Detected {len(sensitive_cols)} sensitive attributes: **{', '.join(sensitive_cols)}**")
                
                st.markdown("---")
                st.markdown("**Select Attribute Combinations to Analyze:**")
                st.caption("Choose which pairs of sensitive attributes you want to analyze together. Only selected combinations will generate visualizations.")
                
                # Generate all possible pairs
                from itertools import combinations
                possible_pairs = list(combinations(sensitive_cols, 2))
                
                # Initialize session state for checkboxes
                if 'combo_selections' not in st.session_state:
                    st.session_state.combo_selections = {}
                
                # Display checkboxes for each combination
                selected_pairs = []
                
                # Create columns for better layout (2 columns)
                num_cols = 2
                cols = st.columns(num_cols)
                
                for idx, (attr1, attr2) in enumerate(possible_pairs):
                    col_idx = idx % num_cols
                    combo_key = f"{attr1}_{attr2}"
                    
                    # Initialize checkbox state if not exists
                    if combo_key not in st.session_state.combo_selections:
                        st.session_state.combo_selections[combo_key] = False
                    
                    with cols[col_idx]:
                        is_selected = st.checkbox(
                            f"{attr1} + {attr2}",
                            value=st.session_state.combo_selections[combo_key],
                            key=f"checkbox_{combo_key}"
                        )
                        
                        # Update state and collect selected pairs
                        st.session_state.combo_selections[combo_key] = is_selected
                        if is_selected:
                            selected_pairs.append((attr1, attr2))
                
                # Show selection summary and generate button
                st.markdown("---")
                
                if selected_pairs:
                    st.success(f"**{len(selected_pairs)} combination(s) selected:** {', '.join([f'{a}+{b}' for a, b in selected_pairs])}")
                else:
                    st.warning("No combinations selected. Please select at least one combination above.")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    if st.button("Clear All", key="clear_combos", use_container_width=True):
                        st.session_state.combo_selections = {k: False for k in st.session_state.combo_selections.keys()}
                        st.rerun()
                
                with col3:
                    if st.button("Generate Analysis", type="primary", key="gen_stage_4_5", use_container_width=True, disabled=len(selected_pairs) == 0):
                        # Execute stage with selected pairs
                        with st.spinner(f"Running {stage_name} for {len(selected_pairs)} combination(s)..."):
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
                                
                                # Clear selections after successful generation
                                st.session_state.combo_selections = {}
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error in {stage_name}: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                                return
                
                return  # Don't show continue button yet
        
        # Special handling for Stage 6 - Bias Mitigation
        if stage_key == "6_bias_mitigation" and stage_key not in results["stages"]:
            st.markdown("### Stage 6: Bias Mitigation")
            st.info("Apply bias mitigation techniques to generate balanced datasets")
            
            # Check if we should show the mitigation options
            if 'show_mitigation_prompt' not in st.session_state:
                st.session_state.show_mitigation_prompt = True
            
            if st.session_state.show_mitigation_prompt:
                st.markdown("---")
                st.markdown("**Would you like to apply bias mitigation techniques to fix imbalances?**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Yes, apply mitigation", type="primary", key="yes_mitigation"):
                        st.session_state.show_mitigation_prompt = False
                        st.session_state.apply_mitigation = True
                        st.rerun()
                with col2:
                    if st.button("No, skip this step", key="no_mitigation"):
                        # Create a result indicating skip
                        results["stages"]["6_bias_mitigation"] = {
                            "status": "skipped",
                            "message": "User chose to skip bias mitigation"
                        }
                        st.session_state.show_mitigation_prompt = False
                        st.rerun()
                with col3:
                    pass
                
                return  # Don't proceed until user makes a choice
            
            # If user chose to apply mitigation, show the options
            if st.session_state.get('apply_mitigation', False):
                st.markdown("---")
                st.markdown("#### Select Bias Mitigation Methods")
                
                # Multi-method selection
                selected_methods = st.multiselect(
                    "Select one or more mitigation techniques:",
                    options=["Reweighting", "SMOTE", "Random Oversampling", "Random Undersampling"],
                    default=[],
                    help="You can select multiple methods to compare their effectiveness",
                    key="mitigation_methods"
                )
                
                if not selected_methods:
                    st.info("Please select at least one mitigation method to proceed.")
                    if st.button("Cancel", key="cancel_mitigation_early"):
                        st.session_state.show_mitigation_prompt = True
                        st.session_state.apply_mitigation = False
                        st.rerun()
                    return
                
                # Get sensitive columns
                sensitive_cols = []
                if "3_sensitive" in results["stages"]:
                    sensitive_cols = results["stages"]["3_sensitive"].get("sensitive_columns", [])
                
                # Configure parameters for each selected method
                st.markdown("#### Configure Methods")
                
                all_methods_config = {}
                
                for method in selected_methods:
                    with st.expander(f"{method} Parameters", expanded=True):
                        method_params = {}
                        
                        if method == "Reweighting":
                            st.info("Reweighting assigns sample weights to balance underrepresented groups. No new samples are added.")
                            if sensitive_cols:
                                selected_sensitive = st.multiselect(
                                    "Select sensitive columns for reweighting:",
                                    options=sensitive_cols,
                                    default=sensitive_cols,
                                    help="Weights will be computed based on these sensitive attributes",
                                    key=f"reweight_sensitive_{method}"
                                )
                                method_params['sensitive_columns'] = selected_sensitive
                            else:
                                st.warning("No sensitive columns detected. Reweighting requires sensitive attributes.")
                        
                        elif method == "SMOTE":
                            st.info("SMOTE generates synthetic samples for minority classes using k-nearest neighbors.")
                            col1, col2 = st.columns(2)
                            with col1:
                                k_neighbors = st.number_input(
                                    "K Neighbors:",
                                    min_value=1,
                                    max_value=10,
                                    value=5,
                                    help="Number of nearest neighbors to use for generating synthetic samples",
                                    key=f"smote_k_{method}"
                                )
                                method_params['k_neighbors'] = k_neighbors
                            with col2:
                                sampling_strategy = st.selectbox(
                                    "Sampling Strategy:",
                                    options=["auto", "minority", "not majority", "all"],
                                    help="Which classes to oversample",
                                    key=f"smote_strategy_{method}"
                                )
                                method_params['sampling_strategy'] = sampling_strategy
                        
                        elif method == "Random Oversampling":
                            st.info("Random Oversampling duplicates samples from minority classes.")
                            sampling_strategy = st.selectbox(
                                "Sampling Strategy:",
                                options=["auto", "minority", "not majority", "all"],
                                help="Which classes to oversample",
                                key=f"oversample_strategy_{method}"
                            )
                            method_params['sampling_strategy'] = sampling_strategy
                        
                        elif method == "Random Undersampling":
                            st.info("Random Undersampling removes samples from majority classes.")
                            st.warning("This method reduces dataset size")
                            sampling_strategy = st.selectbox(
                                "Sampling Strategy:",
                                options=["auto", "not minority", "majority", "all"],
                                help="Which classes to undersample",
                                key=f"undersample_strategy_{method}"
                            )
                            method_params['sampling_strategy'] = sampling_strategy
                        
                        all_methods_config[method] = method_params
                
                st.markdown("---")
                
                # Apply button
                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    if st.button("Cancel", key="cancel_mitigation"):
                        st.session_state.show_mitigation_prompt = True
                        st.session_state.apply_mitigation = False
                        st.rerun()
                
                with col3:
                    # Check if all methods have valid parameters
                    can_apply = True
                    for method in selected_methods:
                        if method == "Reweighting" and not all_methods_config[method].get('sensitive_columns'):
                            can_apply = False
                            break
                    
                    if st.button("Apply All Methods", type="primary", key="apply_all_btn", disabled=not can_apply):
                        # Immediately mark stage as in-progress to prevent re-execution
                        if "6_bias_mitigation" not in results["stages"]:
                            results["stages"]["6_bias_mitigation"] = {
                                "status": "in_progress",
                                "message": "Processing..."
                            }
                        
                        # Map method names to internal names
                        method_map = {
                            "Reweighting": "reweighting",
                            "SMOTE": "smote",
                            "Random Oversampling": "oversampling",
                            "Random Undersampling": "undersampling"
                        }
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        all_results = {}
                        
                        for idx, method in enumerate(selected_methods):
                            status_text.text(f"Applying {method}... ({idx + 1}/{len(selected_methods)})")
                            
                            try:
                                method_params = all_methods_config[method]
                                
                                # Apply mitigation
                                mitigation_result = pipeline.apply_bias_mitigation(
                                    method=method_map[method],
                                    dataset_name=results.get('dataset'),
                                    target_column=results.get('target_column'),
                                    sensitive_columns=method_params.get('sensitive_columns'),
                                    **{k: v for k, v in method_params.items() if k != 'sensitive_columns'}
                                )
                                
                                # Compare results if successful
                                if mitigation_result.get("status") == "success":
                                    comparison_result = pipeline.compare_mitigation_results(
                                        original_dataset=results.get('dataset'),
                                        mitigated_dataset=mitigation_result.get("output_file"),
                                        target_column=results.get('target_column'),
                                        sensitive_columns=sensitive_cols
                                    )
                                    
                                    all_results[method] = {
                                        "status": "success",
                                        "method_params": method_params,
                                        "mitigation_result": mitigation_result,
                                        "comparison_result": comparison_result
                                    }
                                else:
                                    all_results[method] = {
                                        "status": "error",
                                        "error": mitigation_result.get("message", "Unknown error")
                                    }
                                
                            except Exception as e:
                                all_results[method] = {
                                    "status": "error",
                                    "error": str(e)
                                }
                            
                            progress_bar.progress((idx + 1) / len(selected_methods))
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Store all results
                        results["stages"]["6_bias_mitigation"] = {
                            "status": "success",
                            "methods": all_results,
                            "applied_methods": selected_methods
                        }
                        
                        st.session_state.apply_mitigation = False
                        st.success(f"✓ Successfully applied {len(selected_methods)} method(s)!")
                        st.rerun()
                
                return 
        
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
    if stage_key == "0_loading":
        return pipeline._stage_0_load_dataset(dataset_name)
    elif stage_key == "1_objective":
        return pipeline._stage_1_objective_inspection(user_prompt)
    elif stage_key == "2_quality":
        return pipeline._stage_2_data_quality(dataset_name)
    elif stage_key == "3_sensitive":
        return pipeline._stage_3_sensitive_detection(dataset_name, target_column)
    elif stage_key == "4_imbalance":
        return pipeline._stage_4_imbalance_analysis(dataset_name)
    elif stage_key == "4_5_target_fairness":
        return pipeline._stage_4_5_target_fairness_analysis(dataset_name, target_column)
    elif stage_key == "5_recommendations":
        return pipeline._stage_6_recommendations()
    else:
        return {"status": "error", "message": f"Unknown stage: {stage_key}"}

def execute_stage_with_pairs(pipeline, stage_key, user_prompt, dataset_name, target_column, selected_pairs):
    if stage_key == "4_5_target_fairness":
        return pipeline._stage_4_5_target_fairness_analysis(dataset_name, target_column, selected_pairs)
    else:
        return execute_stage(pipeline, stage_key, user_prompt, dataset_name, target_column)

def display_stage_results(stage_key, stage_result):
    # Extract stage name from key
    stage_names = {
        "0_loading": "Stage 0: Dataset Loading",
        "1_objective": "Stage 1: Objective Inspection",
        "2_quality": "Stage 2: Data Quality Analysis",
        "3_sensitive": "Stage 3: Sensitive Attribute Detection",
        "4_imbalance": "Stage 4: Imbalance Analysis",
        "4_5_target_fairness": "Stage 4.5: Target Fairness Analysis",
        "5_recommendations": "Stage 5: Recommendations",
        "6_bias_mitigation": "Stage 6: Bias Mitigation"
    }
    
    st.markdown(f"### {stage_names.get(stage_key, stage_key)}")
    
    # Special display for Stage 6 (Bias Mitigation)
    if stage_key == "6_bias_mitigation":
        if stage_result.get("status") == "skipped":
            st.info("Bias mitigation was skipped by user.")
            return
        elif stage_result.get("status") == "error":
            st.error(f"Error applying {stage_result.get('method', 'mitigation')}: {stage_result.get('error', 'Unknown error')}")
            return
        elif stage_result.get("status") == "success":
            # Check if multiple methods were applied
            if "methods" in stage_result:
                # Multi-method display
                methods_results = stage_result["methods"]
                applied_methods = stage_result.get("applied_methods", list(methods_results.keys()))
                
                st.success(f"✓ Successfully applied {len(applied_methods)} method(s)!")
                
                # Create comparison dashboard first
                st.markdown("---")
                st.markdown("### Methods Comparison Dashboard")
                
                import pandas as pd
                
                # Collect comparison metrics
                comparison_data = []
                successful_methods = {}
                
                for method in applied_methods:
                    method_result = methods_results.get(method, {})
                    if method_result.get("status") == "success":
                        successful_methods[method] = method_result
                        
                        mitigation_result = method_result.get("mitigation_result", {})
                        comparison_result = method_result.get("comparison_result", {})
                        imb_metrics = comparison_result.get("imbalance_metrics", {})
                        
                        original_rows = mitigation_result.get("original_rows", 0)
                        new_rows = mitigation_result.get("new_rows", original_rows)
                        rows_change = new_rows - original_rows
                        
                        orig_ratio = imb_metrics.get("original_imbalance_ratio", 0)
                        mit_ratio = imb_metrics.get("mitigated_imbalance_ratio", 0)
                        improved = imb_metrics.get("improvement", "No")
                        
                        comparison_data.append({
                            "Method": method,
                            "Original Rows": f"{original_rows:,}",
                            "New Rows": f"{new_rows:,}",
                            "Rows Change": f"{rows_change:+,}" if rows_change != 0 else "0",
                            "Original Imbalance": f"{orig_ratio:.2f}",
                            "New Imbalance": f"{mit_ratio:.2f}",
                            "Improvement": "✓" if improved == "Yes" else "✗"
                        })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                    
                    # Best method recommendation
                    best_method = None
                    best_ratio = float('inf')
                    for method in applied_methods:
                        method_result = methods_results.get(method, {})
                        if method_result.get("status") == "success":
                            comparison_result = method_result.get("comparison_result", {})
                            imb_metrics = comparison_result.get("imbalance_metrics", {})
                            mit_ratio = imb_metrics.get("mitigated_imbalance_ratio", float('inf'))
                            if mit_ratio < best_ratio:
                                best_ratio = mit_ratio
                                best_method = method
                    
                    if best_method:
                        st.info(f"**Best Method:** {best_method} achieved the lowest imbalance ratio ({best_ratio:.2f})")
                else:
                    st.warning("No successful methods to compare.")
                
                # Individual method results in dropdowns
                st.markdown("---")
                st.markdown("### Individual Method Results")
                
                for method in applied_methods:
                    method_result = methods_results.get(method, {})
                    
                    with st.expander(f"{method} - Detailed Results", expanded=False):
                        if method_result.get("status") == "error":
                            st.error(f"Error: {method_result.get('error', 'Unknown error')}")
                            continue
                        
                        mitigation_result = method_result.get("mitigation_result", {})
                        comparison_result = method_result.get("comparison_result", {})
                        
                        # Summary metrics
                        st.markdown("#### Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            original_rows = mitigation_result.get("original_rows", 0)
                            st.metric("Original Rows", f"{original_rows:,}")
                        
                        with col2:
                            new_rows = mitigation_result.get("new_rows", original_rows)
                            st.metric("New Rows", f"{new_rows:,}")
                        
                        with col3:
                            if "rows_added" in mitigation_result:
                                st.metric("Rows Added", f"+{mitigation_result['rows_added']:,}", delta=mitigation_result['rows_added'])
                            elif "rows_removed" in mitigation_result:
                                st.metric("Rows Removed", f"-{mitigation_result['rows_removed']:,}", delta=-mitigation_result['rows_removed'])
                        
                        with col4:
                            output_file = mitigation_result.get("output_file", "")
                            if output_file:
                                filename = os.path.basename(output_file)
                                st.metric("Output File", "✓")
                                st.caption(filename)
                        
                        # Distribution comparison
                        st.markdown("---")
                        st.markdown("#### Target Distribution Comparison")
                        
                        dist_before = mitigation_result.get("distribution_before", {})
                        dist_after = mitigation_result.get("distribution_after", dist_before)
                        
                        if dist_before and dist_after:
                            # Create comparison dataframe
                            all_values = set(dist_before.keys()) | set(dist_after.keys())
                            dist_comparison_data = []
                            
                            for value in sorted(all_values):
                                before_count = dist_before.get(value, 0)
                                after_count = dist_after.get(value, 0)
                                before_pct = (before_count / sum(dist_before.values()) * 100) if dist_before else 0
                                after_pct = (after_count / sum(dist_after.values()) * 100) if dist_after else 0
                                
                                dist_comparison_data.append({
                                    "Class": str(value),
                                    "Before Count": before_count,
                                    "Before %": f"{before_pct:.2f}%",
                                    "After Count": after_count,
                                    "After %": f"{after_pct:.2f}%",
                                    "Change": after_count - before_count
                                })
                            
                            df_dist = pd.DataFrame(dist_comparison_data)
                            st.dataframe(df_dist, use_container_width=True, hide_index=True)
                        
                        # Imbalance metrics
                        if comparison_result.get("imbalance_metrics"):
                            st.markdown("---")
                            st.markdown("#### Imbalance Improvement")
                            
                            imb_metrics = comparison_result["imbalance_metrics"]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                orig_ratio = imb_metrics.get("original_imbalance_ratio", 0)
                                st.metric("Original Imbalance Ratio", f"{orig_ratio:.2f}")
                            
                            with col2:
                                mit_ratio = imb_metrics.get("mitigated_imbalance_ratio", 0)
                                delta = mit_ratio - orig_ratio
                                st.metric("Mitigated Imbalance Ratio", f"{mit_ratio:.2f}", 
                                         delta=f"{delta:.2f}", delta_color="inverse")
                            
                            with col3:
                                improved = imb_metrics.get("improvement", "No")
                                if improved == "Yes":
                                    st.success("Imbalance Improved")
                                else:
                                    st.warning("No Improvement")
                        
                        # Agent analysis
                        if comparison_result.get("agent_analysis"):
                            st.markdown("---")
                            st.markdown("#### Agent Analysis")
                            st.markdown(comparison_result["agent_analysis"])
                        
                        # Download button for generated CSV
                        if output_file and os.path.exists(output_file):
                            st.markdown("---")
                            with open(output_file, 'rb') as f:
                                st.download_button(
                                    label=f"Download {method} Dataset",
                                    data=f,
                                    file_name=os.path.basename(output_file),
                                    mime="text/csv",
                                    key=f"download_{method.replace(' ', '_')}"
                                )
            else:
                # Single method display (backward compatibility)
                st.success(f"{stage_result.get('method', 'Bias mitigation')} applied successfully!")
                
                # Display mitigation results
                mitigation_result = stage_result.get("mitigation_result", {})
                comparison_result = stage_result.get("comparison_result", {})
                
                # Summary metrics
                st.markdown("#### Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    original_rows = mitigation_result.get("original_rows", 0)
                    st.metric("Original Rows", f"{original_rows:,}")
                
                with col2:
                    new_rows = mitigation_result.get("new_rows", original_rows)
                    st.metric("New Rows", f"{new_rows:,}")
                
                with col3:
                    if "rows_added" in mitigation_result:
                        st.metric("Rows Added", f"+{mitigation_result['rows_added']:,}", delta=mitigation_result['rows_added'])
                    elif "rows_removed" in mitigation_result:
                        st.metric("Rows Removed", f"-{mitigation_result['rows_removed']:,}", delta=-mitigation_result['rows_removed'])
                
                with col4:
                    output_file = mitigation_result.get("output_file", "")
                    if output_file:
                        filename = os.path.basename(output_file)
                        st.metric("Output File", "✓")
                        st.caption(filename)
                
                # Distribution comparison
                st.markdown("---")
                st.markdown("#### Target Distribution Comparison")
                
                dist_before = mitigation_result.get("distribution_before", {})
                dist_after = mitigation_result.get("distribution_after", dist_before)
                
                if dist_before and dist_after:
                    import pandas as pd
                    
                    # Create comparison dataframe
                    all_values = set(dist_before.keys()) | set(dist_after.keys())
                    comparison_data = []
                    
                    for value in sorted(all_values):
                        before_count = dist_before.get(value, 0)
                        after_count = dist_after.get(value, 0)
                        before_pct = (before_count / sum(dist_before.values()) * 100) if dist_before else 0
                        after_pct = (after_count / sum(dist_after.values()) * 100) if dist_after else 0
                        
                        comparison_data.append({
                            "Class": str(value),
                            "Before Count": before_count,
                            "Before %": f"{before_pct:.2f}%",
                            "After Count": after_count,
                            "After %": f"{after_pct:.2f}%",
                            "Change": after_count - before_count
                        })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
                # Imbalance metrics
                if comparison_result.get("imbalance_metrics"):
                    st.markdown("---")
                    st.markdown("#### Imbalance Improvement")
                    
                    imb_metrics = comparison_result["imbalance_metrics"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        orig_ratio = imb_metrics.get("original_imbalance_ratio", 0)
                        st.metric("Original Imbalance Ratio", f"{orig_ratio:.2f}")
                    
                    with col2:
                        mit_ratio = imb_metrics.get("mitigated_imbalance_ratio", 0)
                        delta = mit_ratio - orig_ratio
                        st.metric("Mitigated Imbalance Ratio", f"{mit_ratio:.2f}", 
                                 delta=f"{delta:.2f}", delta_color="inverse")
                    
                    with col3:
                        improved = imb_metrics.get("improvement", "No")
                        if improved == "Yes":
                            st.success("Imbalance Improved")
                        else:
                            st.warning("No Improvement")
                
                # Agent analysis
                if comparison_result.get("agent_analysis"):
                    st.markdown("---")
                    st.markdown("#### Agent Analysis")
                    with st.expander("View Detailed Analysis", expanded=True):
                        st.markdown(comparison_result["agent_analysis"])
                
                # Download button for generated CSV
                if output_file and os.path.exists(output_file):
                    st.markdown("---")
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            label="Download Mitigated Dataset",
                            data=f,
                            file_name=os.path.basename(output_file),
                            mime="text/csv",
                            key="download_mitigated"
                        )
            
            return
    
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
    
    # Special display for Stage 5 (Recommendations)
    if stage_key == "5_recommendations" and "recommendations" in stage_result:
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
                
                # Display main/simple visualizations with selectbox
                if main_images:
                    st.markdown("#### Main Visualizations")
                    
                    # Create list of image options
                    main_image_options = {}
                    for img_path in main_images:
                        if os.path.exists(img_path):
                            filename = os.path.basename(img_path)
                            display_name = filename.replace('.png', '').replace('_', ' ').title()
                            main_image_options[display_name] = img_path
                    
                    if main_image_options:
                        selected_main = st.selectbox(
                            "Select visualization to view:",
                            options=["None"] + list(main_image_options.keys()),
                            key="main_viz_selector"
                        )
                        
                        if selected_main != "None":
                            st.image(main_image_options[selected_main], caption=selected_main, use_container_width=True)
                
                # Display combination visualizations with better selection
                if combination_images:
                    st.markdown("---")
                    st.markdown("#### Combined Sensitive Attribute Analysis")
                    
                    # Count total combination charts
                    total_charts = sum(len(imgs) for imgs in combination_images.values())
                    st.info(f"{len(combination_images)} attribute combinations available ({total_charts} total charts)")
                    
                    # Let user select which combination to view
                    combo_options = sorted(combination_images.keys())
                    selected_combo = st.selectbox(
                        "Select attribute combination to analyze:",
                        options=["None"] + combo_options,
                        help="Choose which combination of sensitive attributes you want to analyze",
                        key="combo_selector"
                    )
                    
                    if selected_combo != "None":
                        st.markdown(f"##### {selected_combo}")
                        combo_imgs = combination_images[selected_combo]
                        
                        # Create list of images for this combination
                        combo_image_options = {}
                        for img_path in combo_imgs:
                            if os.path.exists(img_path):
                                filename = os.path.basename(img_path)
                                
                                # Create display name
                                if 'scale.png' in filename:
                                    display_name = filename.replace('_scale.png', '').upper() + " Scale"
                                elif 'individual_combinations' in img_path:
                                    display_name = filename.replace('.png', '').replace('_', ' - ')
                                else:
                                    display_name = filename.replace('.png', '').replace('_', ' ').title()
                                
                                combo_image_options[display_name] = img_path
                        
                        if combo_image_options:
                            selected_combo_img = st.selectbox(
                                f"Select {selected_combo} visualization:",
                                options=["None"] + list(combo_image_options.keys()),
                                key=f"combo_img_selector_{selected_combo.replace(' + ', '_')}"
                            )
                            
                            if selected_combo_img != "None":
                                st.image(combo_image_options[selected_combo_img], 
                                       caption=selected_combo_img, 
                                       use_container_width=True)
                else:
                    st.warning("No combination visualizations were generated (requires at least 2 sensitive attributes)")
            
            # Show agent analysis for Stage 4.5
            with st.expander("Agent Analysis", expanded=True):
                st.markdown(stage_result["agent_analysis"])
        
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

def parse_report_file(filepath):
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by stage headers
    stages = {}
    
    # Extract header information
    lines = content.split('\n')
    header_info = {}
    for i, line in enumerate(lines[:30]):  # Check first 30 lines
        if 'Dataset:' in line:
            header_info['dataset'] = line.split('Dataset:')[1].strip()
        elif 'Timestamp:' in line:
            header_info['timestamp'] = line.split('Timestamp:')[1].strip()
        elif 'Target Column:' in line:
            header_info['target'] = line.split('Target Column:')[1].strip()
        elif 'User Objective:' in line:
            header_info['objective'] = line.split('User Objective:')[1].strip()
    
    # Split by STAGE markers
    stage_pattern = r'STAGE\s+(\d+(?:\.\d+)?)[:\s]+(.*?)(?=\n-{5,})'
    import re
    
    current_stage = None
    current_content = []
    
    for line in lines:
        # Check if this is a stage header
        stage_match = re.match(r'STAGE\s+(\d+(?:\.\d+)?)[:\s]+(.*)', line)
        
        if stage_match:
            # Save previous stage
            if current_stage:
                stages[current_stage] = '\n'.join(current_content).strip()
            
            # Start new stage
            stage_num = stage_match.group(1)
            stage_name = stage_match.group(2).strip()
            current_stage = f"Stage {stage_num}: {stage_name}"
            current_content = []
        
        elif line.startswith('=' * 10) or line.startswith('-' * 10):
            # Skip separator lines
            continue
        
        elif current_stage:
            current_content.append(line)
    
    # Save last stage
    if current_stage:
        stages[current_stage] = '\n'.join(current_content).strip()
    
    return header_info, stages

def display_parsed_report(filepath, report_type="Full Report"):
    result = parse_report_file(filepath)
    
    if not result:
        st.warning(f"{report_type} file not found")
        return
    
    header_info, stages = result
    
    # Display header information
    if header_info:
        with st.container():
            st.markdown("### Report Information")
            cols = st.columns(3)
            if 'dataset' in header_info:
                cols[0].metric("Dataset", header_info['dataset'])
            if 'timestamp' in header_info:
                cols[1].metric("Timestamp", header_info['timestamp'])
            if 'target' in header_info:
                cols[2].metric("Target Column", header_info['target'])
            
            if 'objective' in header_info:
                st.info(f"**Objective:** {header_info['objective']}")
        
        st.markdown("---")
    
    # Display stages in expandable sections
    if stages:
        st.markdown("### Report Stages")
        for stage_name, stage_content in stages.items():
            with st.expander(stage_name, expanded=False):
                # Check if content has subsections
                if '[TOOL USED]' in stage_content or '[AGENT ANALYSIS]' in stage_content:
                    # Split into subsections
                    sections = stage_content.split('[TOOL USED]')
                    
                    for section in sections:
                        if not section.strip():
                            continue
                        
                        if '[TOOL RESULT]' in section and '[AGENT ANALYSIS]' in section:
                            # Split tool result and analysis
                            parts = section.split('[TOOL RESULT]')
                            tool_name = parts[0].strip()
                            
                            remaining = parts[1].split('[AGENT ANALYSIS]')
                            tool_result = remaining[0].strip()
                            agent_analysis = remaining[1].strip() if len(remaining) > 1 else ""
                            
                            if tool_name:
                                st.markdown(f"**Tool Used:** `{tool_name}`")
                            
                            if tool_result:
                                with st.expander("Tool Result", expanded=False):
                                    st.code(tool_result, language='json')
                            
                            if agent_analysis:
                                st.markdown("**Agent Analysis:**")
                                st.markdown(agent_analysis)
                        
                        elif '[AGENT ANALYSIS]' in section:
                            parts = section.split('[AGENT ANALYSIS]')
                            st.markdown("**Agent Analysis:**")
                            st.markdown(parts[1].strip())
                        
                        else:
                            st.markdown(section.strip())
                
                elif '[RECOMMENDATIONS]' in stage_content:
                    # Special handling for recommendations
                    parts = stage_content.split('[RECOMMENDATIONS]')
                    st.markdown(parts[1].strip() if len(parts) > 1 else stage_content)
                
                else:
                    # Simple content
                    st.markdown(stage_content)
    else:
        st.warning("No stages found in report")

def view_results_page():
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
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Full Report", "Agent Summary", "Recommendations", "Bias Mitigation", "Visualizations"])
        
        with tab1:
            display_parsed_report(report_file, "Full Report")
        
        with tab2:
            display_parsed_report(summary_file, "Agent Summary")
        
        with tab3:
            # Display Stage 5 - Recommendations
            st.markdown("### Stage 5: Recommendations")
            if os.path.exists(report_file):
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Try to extract recommendations section - look for multiple possible headers
                    rec_start = -1
                    rec_end = -1
                    
                    # Try different header formats
                    if "[RECOMMENDATIONS]" in content:
                        rec_start = content.find("[RECOMMENDATIONS]")
                        # Find the next major section (Stage or end marker)
                        temp = content[rec_start:]
                        # Look for next stage marker
                        next_markers = [
                            temp.find("\n\n6_BIAS_MITIGATION"),
                            temp.find("\n\nSTAGE 6:"),
                            temp.find("\n\n================================================================================\nEND OF REPORT"),
                        ]
                        next_markers = [m for m in next_markers if m > 0]
                        if next_markers:
                            rec_end = rec_start + min(next_markers)
                    elif "5_RECOMMENDATIONS" in content:
                        rec_start = content.find("5_RECOMMENDATIONS")
                        # Find the next stage
                        temp = content[rec_start:]
                        next_markers = [
                            temp.find("\n\n6_BIAS_MITIGATION"),
                            temp.find("\n\nSTAGE 6:"),
                            temp.find("\n\n================================================================================\nEND OF REPORT"),
                        ]
                        next_markers = [m for m in next_markers if m > 0]
                        if next_markers:
                            rec_end = rec_start + min(next_markers)
                    
                    if rec_start >= 0:
                        if rec_end > rec_start:
                            rec_section = content[rec_start:rec_end]
                        else:
                            rec_section = content[rec_start:]
                        st.markdown(rec_section)
                    else:
                        st.info("No recommendations found in this report.")
            else:
                st.info("Report file not found.")
        
        with tab4:
            # Display Stage 6 - Bias Mitigation results
            st.markdown("### Stage 6: Bias Mitigation Results")
            
            # Parse agent analysis from report
            report_file = os.path.join(report_dir, "evaluation_report.txt")
            methods_analysis = {}
            
            if os.path.exists(report_file):
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for 6_BIAS_MITIGATION section
                if "6_BIAS_MITIGATION" in content:
                    # Extract the entire bias mitigation section
                    bias_start = content.find("\n\n6_BIAS_MITIGATION")
                    if bias_start >= 0:
                        # Find the end of this section (next major section or END OF REPORT)
                        temp = content[bias_start:]
                        end_markers = [
                            temp.find("\n\n================================================================================\nEND OF REPORT"),
                        ]
                        end_markers = [m for m in end_markers if m > 0]
                        
                        if end_markers:
                            bias_end = bias_start + min(end_markers)
                            bias_section = content[bias_start:bias_end]
                        else:
                            bias_section = temp
                        
                        # Parse each method's analysis
                        # Methods are marked with [METHOD_NAME]
                        import re
                        method_pattern = r'\[([A-Z][A-Z\s]+)\]\n-{40}'
                        method_matches = list(re.finditer(method_pattern, bias_section))
                        
                        for i, match in enumerate(method_matches):
                            method_name = match.group(1).strip()
                            method_start = match.end()
                            
                            # Find the next method or end of section
                            if i + 1 < len(method_matches):
                                method_end = method_matches[i + 1].start()
                            else:
                                method_end = len(bias_section)
                            
                            method_content = bias_section[method_start:method_end]
                            
                            # Extract agent analysis
                            if "[AGENT ANALYSIS]" in method_content:
                                analysis_start = method_content.find("[AGENT ANALYSIS]") + len("[AGENT ANALYSIS]")
                                analysis_text = method_content[analysis_start:].strip()
                                
                                # Clean up - remove any following section markers
                                if "\n[" in analysis_text:
                                    analysis_text = analysis_text[:analysis_text.find("\n[")].strip()
                                
                                methods_analysis[method_name] = analysis_text
            
            # Check for generated CSV files
            generated_csv_dir = os.path.join(report_dir, "generated_csv")
            if os.path.exists(generated_csv_dir):
                csv_files = [f for f in os.listdir(generated_csv_dir) if f.endswith('.csv')]
                
                if csv_files:
                    st.success(f"Found {len(csv_files)} mitigated dataset(s)")
                    
                    # Try to parse method names from filenames
                    methods_data = {}
                    for csv_file in csv_files:
                        if 'smote' in csv_file.lower():
                            methods_data['SMOTE'] = csv_file
                        elif 'reweighted' in csv_file.lower():
                            methods_data['Reweighting'] = csv_file
                        elif 'oversampled' in csv_file.lower():
                            methods_data['Random Oversampling'] = csv_file
                        elif 'undersampled' in csv_file.lower():
                            methods_data['Random Undersampling'] = csv_file
                    
                    if methods_data:
                        # Create comparison table
                        st.markdown("#### Methods Comparison")
                        
                        import pandas as pd
                        comparison_data = []
                        
                        for method, filename in methods_data.items():
                            filepath = os.path.join(generated_csv_dir, filename)
                            try:
                                df = pd.read_csv(filepath)
                                row_count = len(df)
                                
                                # Try to get target column (assume it's mentioned in report)
                                # For now, just show basic stats
                                comparison_data.append({
                                    "Method": method,
                                    "Rows": f"{row_count:,}",
                                    "File": filename
                                })
                            except Exception as e:
                                st.error(f"Error reading {filename}: {str(e)}")
                        
                        if comparison_data:
                            df_comparison = pd.DataFrame(comparison_data)
                            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                        
                        # Show individual method results in dropdowns
                        st.markdown("---")
                        st.markdown("#### Individual Method Details")
                        
                        for method, filename in methods_data.items():
                            with st.expander(f"{method} - Detailed Results"):
                                filepath = os.path.join(generated_csv_dir, filename)
                                
                                try:
                                    df = pd.read_csv(filepath)
                                    
                                    st.markdown("##### Dataset Information")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Rows", f"{len(df):,}")
                                    with col2:
                                        st.metric("Total Columns", len(df.columns))
                                    with col3:
                                        if 'sample_weight' in df.columns:
                                            st.metric("Has Weights", "Yes")
                                        else:
                                            st.metric("Has Weights", "No")
                                    
                                    st.markdown("##### Column Names")
                                    st.write(", ".join(df.columns.tolist()))
                                    
                                    st.markdown("##### Sample Data (First 5 Rows)")
                                    st.dataframe(df.head(), use_container_width=True)
                                    
                                    # Display agent analysis if available
                                    method_upper = method.upper()
                                    if method_upper in methods_analysis:
                                        st.markdown("---")
                                        st.markdown("##### Agent Analysis")
                                        st.markdown(methods_analysis[method_upper])
                                    elif "RANDOM OVERSAMPLING" in methods_analysis and method == "Random Oversampling":
                                        st.markdown("---")
                                        st.markdown("##### Agent Analysis")
                                        st.markdown(methods_analysis["RANDOM OVERSAMPLING"])
                                    elif "RANDOM UNDERSAMPLING" in methods_analysis and method == "Random Undersampling":
                                        st.markdown("---")
                                        st.markdown("##### Agent Analysis")
                                        st.markdown(methods_analysis["RANDOM UNDERSAMPLING"])
                                    
                                    # Download button
                                    st.markdown("---")
                                    with open(filepath, 'rb') as f:
                                        st.download_button(
                                            label=f"Download {method} Dataset",
                                            data=f,
                                            file_name=filename,
                                            mime="text/csv",
                                            key=f"download_prev_{method.replace(' ', '_')}"
                                        )
                                    
                                except Exception as e:
                                    st.error(f"Error displaying {filename}: {str(e)}")
                    else:
                        st.info("Generated CSV files found, but could not identify mitigation methods.")
                else:
                    st.info("No bias mitigation was applied in this evaluation.")
            else:
                st.info("No bias mitigation was applied in this evaluation.")
        
        with tab5:
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
                    
                    # Separate main images from combination images
                    main_images = []
                    combination_images = {}
                    
                    for img_path in image_files:
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
                    
                    # Display main/simple visualizations
                    if main_images:
                        st.markdown("#### Main Visualizations")
                        
                        # Create list of image options
                        main_image_options = {}
                        for img_path in main_images:
                            if os.path.exists(img_path):
                                filename = os.path.basename(img_path)
                                display_name = filename.replace('.png', '').replace('_', ' ').title()
                                main_image_options[display_name] = img_path
                        
                        if main_image_options:
                            selected_main = st.selectbox(
                                "Select visualization to view:",
                                options=["None"] + list(main_image_options.keys()),
                                key="prev_main_viz_selector"
                            )
                            
                            if selected_main != "None":
                                st.image(main_image_options[selected_main], caption=selected_main, use_container_width=True)
                    
                    # Display combination visualizations
                    if combination_images:
                        st.markdown("---")
                        st.markdown("#### Combined Sensitive Attribute Analysis")
                        
                        # Count total combination charts
                        total_charts = sum(len(imgs) for imgs in combination_images.values())
                        st.info(f"{len(combination_images)} attribute combinations available ({total_charts} total charts)")
                        
                        # Let user select which combination to view
                        combo_options = sorted(combination_images.keys())
                        selected_combo = st.selectbox(
                            "Select attribute combination to analyze:",
                            options=["None"] + combo_options,
                            help="Choose which combination of sensitive attributes you want to view",
                            key="prev_combo_selector"
                        )
                        
                        if selected_combo != "None":
                            st.markdown(f"##### {selected_combo}")
                            combo_imgs = combination_images[selected_combo]
                            
                            # Create list of images for this combination
                            combo_image_options = {}
                            for img_path in combo_imgs:
                                if os.path.exists(img_path):
                                    filename = os.path.basename(img_path)
                                    
                                    # Create display name
                                    if 'scale.png' in filename:
                                        display_name = filename.replace('_scale.png', '').upper() + " Scale"
                                    elif 'individual_combinations' in img_path:
                                        display_name = filename.replace('.png', '').replace('_', ' - ')
                                    else:
                                        display_name = filename.replace('.png', '').replace('_', ' ').title()
                                    
                                    combo_image_options[display_name] = img_path
                            
                            if combo_image_options:
                                selected_combo_img = st.selectbox(
                                    f"Select {selected_combo} visualization:",
                                    options=["None"] + list(combo_image_options.keys()),
                                    key=f"prev_combo_img_selector_{selected_combo.replace(' + ', '_')}"
                                )
                                
                                if selected_combo_img != "None":
                                    st.image(combo_image_options[selected_combo_img], 
                                           caption=selected_combo_img, 
                                           use_container_width=True)
                else:
                    st.info("No images found in this report")
            else:
                st.info("No images directory found")

def main():
    init_session_state()
    
    if st.session_state.mode is None:
        main_page()
    elif st.session_state.mode == "new":
        new_evaluation_page()
    elif st.session_state.mode == "view":
        view_results_page()

if __name__ == "__main__":
    main()
