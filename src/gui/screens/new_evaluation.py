import os
import traceback
from datetime import datetime
from itertools import combinations

import streamlit as st

from pipeline import DatasetEvaluationPipeline
from pipeline.stage import NavigationAction, StageStatus

from gui.config import (
    get_config_path,
    get_available_models,
    get_default_model_name,
    get_default_target_column,
    get_default_ml_model,
    get_default_ml_model_params,
    get_default_dataset,
    validate_api_keys,
)
from gui.utils import (
    BASE_DIR,
    get_available_datasets,
    upload_dataset,
    get_dataset_columns,
)
from gui.widgets.stage_display import display_stage_results
from gui.pdf_generator import generate_pdf_bytes
import requests

def _get_ollama_models() -> list[str]:
    """Fetch installed models from local Ollama instance."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return sorted([m["name"] for m in response.json().get("models", [])])
    except Exception:
        pass
    return []


def new_evaluation_page():
    """Render the full page (sidebar + chatbot area)."""

    st.markdown("<div class='main-header'>New Evaluation</div>", unsafe_allow_html=True)

    # Sidebar -- configuration
    with st.sidebar:
        st.markdown("### Configuration")

        if st.button("\u2190 Back to Main"):
            _reset_state()
            st.rerun()

        st.markdown("---")

        # ---- Dataset ----
        st.markdown("#### Dataset")
        datasets = get_available_datasets()

        uploaded_file = st.file_uploader("Upload new dataset", type=["csv"])
        if uploaded_file:
            dataset_name = upload_dataset(uploaded_file)
            st.success(f"Uploaded: {dataset_name}")
            datasets = get_available_datasets()

        default_ds = get_default_dataset()
        default_ds_idx = datasets.index(default_ds) if default_ds in datasets else 0
        selected_dataset = st.selectbox("Select dataset", datasets, index=default_ds_idx)
        st.session_state.dataset_name = selected_dataset

        # ---- Model selection ----
        st.markdown("#### Model Selection")

        st.session_state.use_ollama = st.checkbox("Use Local Model from Ollama", value=False)
        
        if st.session_state.use_ollama:
            ollama_models = _get_ollama_models()
            if ollama_models:
                st.session_state.model_choice = st.selectbox(
                    "Select Ollama Model:",
                    options=ollama_models
                )
            else:
                st.warning(
                    "Ollama is either not running or has no models installed. "
                    "Make sure the Ollama desktop app is open and running in the background!"
                )
                st.session_state.model_choice = None
        else:
            available_models = get_available_models()
            model_names = list(available_models.keys())
    
            if not model_names:
                st.warning("No models defined in config.yml")
                model_names = ["openrouter"]
    
            def _model_label(model_name):
                cfg = available_models.get(model_name, {})
                provider = cfg.get("provider", "unknown")
                model_id = cfg.get("model", "")
                label = f"{model_name}  ({provider}: {model_id})"
                provider_lower = provider.lower()
                if provider_lower == "openrouter":
                    has_key = bool(os.getenv("OPENROUTER_API_KEY"))
                    return f"{label} {'Online' if has_key else 'Offline'}"
                elif provider_lower in ("gemini", "google"):
                    has_key = bool(os.getenv("GOOGLE_API_KEY"))
                    return f"{label} {'Online' if has_key else 'Offline'}"
                return f"{label} Local"
    
            default_name = get_default_model_name()
            default_idx = model_names.index(default_name) if default_name in model_names else 0
    
            st.session_state.model_choice = st.radio(
                "Choose default model:",
                options=model_names,
                index=default_idx,
                format_func=_model_label,
            )
    
            is_valid, error_msg = validate_api_keys(st.session_state.model_choice)
            if not is_valid:
                st.warning("API key missing for this model")

        # ---- Target column ----
        st.markdown("#### Target Column (Optional)")
        cfg_target = get_default_target_column()
        use_target = st.checkbox(
            "Specify target column for fairness analysis",
            value=bool(cfg_target),
        )

        if use_target and selected_dataset:
            columns = get_dataset_columns(selected_dataset)
            if columns:
                target_idx = columns.index(cfg_target) if cfg_target in columns else 0
                st.session_state.target_column = st.selectbox(
                    "Select target column:", columns, index=target_idx
                )
            else:
                st.warning("Could not read dataset columns")
                st.session_state.target_column = None
        else:
            st.session_state.target_column = None

        # ---- ML Model Type ----
        st.markdown("#### ML Model Type")
        ml_models = ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVC"]
        cfg_ml_model = get_default_ml_model()
        ml_idx = ml_models.index(cfg_ml_model) if cfg_ml_model in ml_models else 0
        st.session_state.ml_model_type = st.selectbox(
            "Select ML Model for Analysis:", 
            ml_models, 
            index=ml_idx
        )

        cfg_params = get_default_ml_model_params()
        default_params = cfg_params.get(st.session_state.ml_model_type, {})
        
        st.markdown("##### Model Parameters")
        st.session_state.ml_model_params = {}
        
        if st.session_state.ml_model_type == "Random Forest":
            st.session_state.ml_model_params["n_estimators"] = st.number_input("n_estimators", min_value=1, value=default_params.get("n_estimators", 100))
            max_depth_val = default_params.get("max_depth")
            use_max_depth = st.checkbox("Set max_depth", value=max_depth_val is not None)
            if use_max_depth:
                st.session_state.ml_model_params["max_depth"] = st.number_input("max_depth", min_value=1, value=max_depth_val or 10)
            else:
                st.session_state.ml_model_params["max_depth"] = None
                
        elif st.session_state.ml_model_type == "Gradient Boosting":
            st.session_state.ml_model_params["n_estimators"] = st.number_input("n_estimators", min_value=1, value=default_params.get("n_estimators", 100))
            st.session_state.ml_model_params["learning_rate"] = st.number_input("learning_rate", min_value=0.01, value=default_params.get("learning_rate", 0.1))
            st.session_state.ml_model_params["max_depth"] = st.number_input("max_depth", min_value=1, value=default_params.get("max_depth", 3))
            
        elif st.session_state.ml_model_type == "Logistic Regression":
            st.session_state.ml_model_params["C"] = st.number_input("C (Inverse of regularization)", min_value=0.01, value=default_params.get("C", 1.0))
            st.session_state.ml_model_params["penalty"] = st.selectbox("penalty", ["l2", "none"], index=0 if default_params.get("penalty", "l2") == "l2" else 1)
            
        elif st.session_state.ml_model_type == "SVC":
            st.session_state.ml_model_params["C"] = st.number_input("C (Regularization)", min_value=0.01, value=default_params.get("C", 1.0))
            kernel_list = ["rbf", "linear", "poly", "sigmoid"]
            st.session_state.ml_model_params["kernel"] = st.selectbox("kernel", kernel_list, index=kernel_list.index(default_params.get("kernel", "rbf")))

        # ---- Additional Reporting Options ----
        st.markdown("#### Reporting Options")
        st.session_state.use_humanizer = st.checkbox(
            "Use Humanizer Agent (Convert AI tone to human)", 
            value=False,
            help="Adds a post-processing AI pass to make agent responses read more naturally."
        )
        st.session_state.generate_detailed_report = st.checkbox(
            "Generate Detailed Report", 
            value=False,
            help="Generates an exhaustive markdown/PDF document with comprehensive metrics for all groups."
        )
        st.session_state.generate_executive_summary = st.checkbox(
            "Generate Executive Summary", 
            value=False,
            help="Runs a final synthesis agent to summarize the most critical fairness risks and mitigation outcomes."
        )

        st.markdown("---")

        # ---- Start / Reset buttons ----
        if selected_dataset:
            if not st.session_state.pipeline_started:
                if st.button("Start Evaluation", width="stretch", type="primary"):
                    _initialize_pipeline()
            else:
                col_reset, col_stop = st.columns(2)
                with col_reset:
                    if st.button("Reset Pipeline", width="stretch"):
                        _reset_state()
                        st.rerun()
                with col_stop:
                    if st.button("Stop & Report", width="stretch", type="secondary"):
                        _stop_and_generate_report()

        # ---- Pipeline progress indicator ----
        if st.session_state.pipeline_started and st.session_state.pipeline:
            pipeline = st.session_state.pipeline
            if pipeline.stages:
                st.markdown("---")
                st.markdown("#### Pipeline Progress")
                total = len(pipeline.stages)
                done = sum(1 for s in pipeline.stages if s.is_completed or s.is_skipped)
                st.progress(done / total if total else 0)
                st.caption(f"Stage {min(pipeline.current_stage_index + 1, total)} / {total}")

                for s in pipeline.stages:
                    icon = {
                        StageStatus.COMPLETED: "\u2705",
                        StageStatus.RUNNING: "\u23f3",
                        StageStatus.ERROR: "\u274c",
                        StageStatus.SKIPPED: "\u23ed",
                        StageStatus.NOT_STARTED: "\u2b1c",
                    }.get(s.status, "\u2b1c")
                    st.caption(f"{icon} {s.name}")

    # Main area -- Chat interface
    if st.session_state.pipeline_started:
        _render_chat_interface()
    else:
        st.markdown(
            """
            <div class='info-box'>
            <h3>Configure your evaluation in the sidebar</h3>
            <p>1. Select or upload a dataset</p>
            <p>2. Choose an AI model</p>
            <p>3. Optionally specify a target column</p>
            <p>4. Click "Start Evaluation" to begin</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Pipeline initialisation
def _initialize_pipeline():
    """Create the pipeline, build stages, and seed the first chat message."""
    try:
        
        if not st.session_state.model_choice:
            st.error("No model selected. Cannot start evaluation.")
            st.session_state.pipeline_started = False
            return
            
        if not st.session_state.get("use_ollama", False):
            is_valid, error_msg = validate_api_keys(st.session_state.model_choice)
            if not is_valid:
                st.error(f"{error_msg}")
                st.session_state.pipeline_started = False
                return

        prompt = (
            f"Evaluate the dataset '{st.session_state.dataset_name}' "
            "for data quality and fairness issues."
        )
        if st.session_state.target_column:
            prompt += f" Target: {st.session_state.target_column}."
        prompt += (
            " Provide a detailed report highlighting any problems found "
            "and suggestions for improvement."
        )

        st.session_state.user_prompt = prompt

        with st.spinner("Initializing pipeline..."):
            pipeline = DatasetEvaluationPipeline(
                config_path=get_config_path(),
                # Bypass yaml loader panic: fall back to none temporarily if using an unlisted Ollama model
                default_model=None if st.session_state.get("use_ollama", False) else st.session_state.model_choice,
            )
            
            # Apply UI overrides to the pipeline config
            if st.session_state.get("use_ollama", False) and st.session_state.model_choice:
                # Ensure the ollama model gets inserted into the agent manager config
                model_name = st.session_state.model_choice
                if "models" not in pipeline.agent_manager.config:
                    pipeline.agent_manager.config["models"] = {}
                pipeline.agent_manager.config["models"][model_name] = {
                    "provider": "ollama",
                    "model": model_name
                }
                pipeline.agent_manager.config["default_model"] = model_name
                
                # Since we bypassed it initially, instantiate the newly registered Ollama client
                pipeline.model_client = pipeline.agent_manager.get_client(model_name)

            pipeline.agent_manager.config["use_humanizer"] = st.session_state.get("use_humanizer", False)
            pipeline.agent_manager.config["generate_detailed_report"] = st.session_state.get("generate_detailed_report", False)
            pipeline.agent_manager.config["generate_executive_summary"] = st.session_state.get("generate_executive_summary", False)
            
            # Clear pre-allocated agents and re-initialize so they all hook into the new model client & UI overrides
            pipeline.agent_manager._agents.clear()
            pipeline._initialize_agents()
            
            pipeline.build_stages(
                dataset_name=st.session_state.dataset_name,
                target_column=st.session_state.target_column,
                user_prompt=prompt,
            )

            # Fix report_dir to project root
            pipeline.report_dir = os.path.join(
                BASE_DIR,
                "reports",
                f"{st.session_state.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            pipeline.images_dir = os.path.join(pipeline.report_dir, "images")
            os.makedirs(pipeline.images_dir, exist_ok=True)
            pipeline._pipeline_ctx["report_dir"] = pipeline.report_dir
            pipeline._pipeline_ctx["images_dir"] = pipeline.images_dir
            pipeline.evaluation_results["report_directory"] = pipeline.report_dir

        st.session_state.pipeline = pipeline
        st.session_state.pipeline_started = True
        st.session_state.evaluation_results = pipeline.evaluation_results

        # Seed the chat
        stage_names = ", ".join(s.name for s in pipeline.stages)
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Pipeline initialised with **{len(pipeline.stages)}** stages for "
                    f"dataset **{st.session_state.dataset_name}**.\n\n"
                    f"Stages: {stage_names}\n\n"
                    "Press **Forward** to execute the first stage. "
                    "You can type additional instructions in the text box before sending."
                ),
            }
        ]
        st.rerun()

    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        st.exception(e)
        st.session_state.pipeline_started = False


# Chat interface
def _render_chat_interface():
    """Render the chat log, stage results, and input controls."""

    pipeline = st.session_state.pipeline
    if not pipeline:
        return

    # ---- Chat message history ----
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)
            elif role == "stage_result":
                with st.chat_message("assistant"):
                    stage_key = msg.get("stage_key", "")
                    stage_data = msg.get("stage_data", {})
                    if stage_key and stage_data:
                        display_stage_results(stage_key, stage_data)
                    else:
                        st.markdown(content)
            elif role == "system":
                with st.chat_message("assistant", avatar="\u2699\ufe0f"):
                    st.info(content)

    # ---- Finished banner ----
    results = st.session_state.evaluation_results
    is_finished = pipeline.current_stage_index >= len(pipeline.stages)
    has_partial_report = results and results.get("partial_report", False)
    
    if is_finished:
        st.success("All stages completed!")
        if results and "final_reports_generated" not in results:
            with st.spinner("Generating final reports..."):
                try:
                    pipeline.evaluation_results = results
                    pipeline.generate_report()
                    results["final_reports_generated"] = True
                except Exception as e:
                    st.error(f"Report generation error: {e}")
    
    # Show download button for both completed and partial reports
    if (is_finished or has_partial_report) and results and "report_directory" in results:
        report_dir = results["report_directory"]
        if has_partial_report and not is_finished:
            st.warning("⏹️ Pipeline stopped. Partial report available.")
        st.markdown(f"**Report Directory:** `{report_dir}`")
        
        # PDF Download button - use evaluation_report.md for markdown formatting
        report_path = os.path.join(report_dir, "evaluation_report.md")
        if os.path.exists(report_path):
            try:
                pdf_bytes = generate_pdf_bytes(report_path)
                dataset_name = os.path.basename(report_dir).split("_")[0]
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{dataset_name}_fairness_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_new_eval",
                )
            except Exception as e:
                    st.warning(f"Could not generate PDF: {e}")

    # ---- Input area ----
    st.markdown("---")
    _render_input_area(pipeline)


def _render_input_area(pipeline):
    """Text input + action selector + send button, with inline stage controls."""

    cur = pipeline.current_stage
    if cur and not pipeline.is_finished:
        st.caption(f"Next stage: **{cur.name}** — {cur.description}")
    elif pipeline.is_finished:
        st.caption("Pipeline finished. You can **Repeat** a stage or **Backward** to revisit.")

    # ---- Inline controls for stages that need human guidance ----
    if cur and not pipeline.is_finished:
        _render_stage_controls(pipeline, cur)

    # Determine available actions based on pipeline state
    if pipeline.is_finished:
        action_options = ["Backward", "Repeat"]
    else:
        action_options = ["Forward", "Backward", "Repeat"]

    with st.form("chat_input_form", clear_on_submit=True):
        col_input, col_action, col_send = st.columns([6, 2, 1])

        with col_input:
            user_text = st.text_input(
                "Message",
                placeholder="Type additional context for the agent (optional)...",
                label_visibility="collapsed",
            )

        with col_action:
            action_choice = st.selectbox(
                "Action",
                options=action_options,
                index=0,
                label_visibility="collapsed",
            )

        with col_send:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted:
        _handle_submit(pipeline, user_text, action_choice)


def _render_stage_controls(pipeline, stage):
    """Render inline controls for stages that benefit from human guidance."""

    dataset_name = st.session_state.get("dataset_name", "")
    columns = get_dataset_columns(dataset_name) if dataset_name else []

    # ------------------------------------------------------------------
    # Stage 3 — Sensitive Attribute Detection
    # ------------------------------------------------------------------
    if stage.key == "3_sensitive":
        with st.expander("Sensitive Attribute Detection — configure before running", expanded=True):
            st.caption(
                "**Auto**: the AI detects sensitive columns, then you can review and filter the results.  "
                "**Manual**: you choose the sensitive attributes directly (skips AI detection)."
            )
            mode = st.radio(
                "Detection mode",
                ["auto", "manual"],
                index=0,
                horizontal=True,
                key="inline_sens_mode",
                format_func=lambda m: "Auto (AI detects)" if m == "auto" else "Manual (I choose)",
            )
            if mode == "manual":
                if columns:
                    st.multiselect(
                        "Select sensitive attributes to use:",
                        options=columns,
                        default=[],
                        key="inline_sens_cols",
                        help="These columns will be used directly as sensitive attributes, skipping AI detection.",
                    )
                    if not st.session_state.get("inline_sens_cols"):
                        st.warning("Select at least one column to proceed in manual mode.")
                else:
                    st.info("Load a dataset to see available columns.")
            else:
                st.info(
                    "After AI detection, you will be able to review and select "
                    "which detected attributes to carry forward."
                )
            # Persist choices so _handle_submit can read them
            st.session_state["_stage3_mode"] = mode

    # ------------------------------------------------------------------
    # Stage 3.5 — Discretization of Continuous Sensitive Attributes
    # ------------------------------------------------------------------
    elif stage.key == "3_5_discretization":
        # Get the sensitive columns confirmed after Stage 3
        sensitive_cols = (
            pipeline.evaluation_results.get("stages", {})
            .get("3_sensitive", {})
            .get("sensitive_columns", [])
        )
        # Use the user-refined list if already available
        refined_cols = pipeline._pipeline_ctx.get("confirmed_sensitive_columns") or sensitive_cols

        with st.expander("Discretization — configure before running", expanded=True):
            # ── Which attributes to discretize ──────────────────────────
            if refined_cols:
                st.markdown("**Attributes to discretize**")
                st.caption(
                    "Select which sensitive attributes should be considered for discretization. "
                    "Only continuous (numeric) ones will actually be discretized."
                )
                selected_for_disc = st.multiselect(
                    "Attributes to include in discretization:",
                    options=refined_cols,
                    default=refined_cols,
                    key="inline_disc_attrs",
                    help="Deselect any attribute you do NOT want to discretize.",
                )
                if not selected_for_disc:
                    st.warning("No attributes selected — discretization will be skipped.")
                st.session_state["_stage35_attrs"] = selected_for_disc
            else:
                st.info("No sensitive columns available from Stage 3.")
                st.session_state["_stage35_attrs"] = []

            st.markdown("---")
            # ── Discretization method ───────────────────────────────────
            st.markdown("**Discretization method**")
            st.caption(
                "Choose a binning strategy for the continuous columns."
            )
            method = st.radio(
                "Discretization method",
                ["auto", "equal_width", "equal_frequency"],
                index=0,
                horizontal=True,
                key="inline_disc_method",
                help=(
                    "**auto**: the agent decides bins based on column statistics. "
                    "**equal_width**: bins of equal range. "
                    "**equal_frequency**: bins of equal sample count."
                ),
            )
            if method in ("equal_width", "equal_frequency"):
                st.number_input(
                    "Number of bins:",
                    min_value=2,
                    max_value=50,
                    value=5,
                    step=1,
                    key="inline_disc_bins",
                )
            st.session_state["_stage35_method"] = method

    # ------------------------------------------------------------------
    # Stage 4.5 — Intersectional Pair Evaluation
    # ------------------------------------------------------------------
    elif stage.key == "4_5_target_fairness":
        # Detected sensitive columns from previous stage result
        sens = (
            pipeline.evaluation_results.get("stages", {})
            .get("3_sensitive", {})
            .get("sensitive_columns", [])
        )
        all_cols = columns or sens
        with st.expander("Intersectional Pair Evaluation — configure before running", expanded=True):
            st.caption("Auto: the agent selects pairs (optionally capped). Restricted: you define exact pairs.")
            mode = st.radio(
                "Mode",
                ["auto", "restricted"],
                index=0,
                horizontal=True,
                key="inline_pair_mode",
            )
            if mode == "auto":
                use_cap = st.checkbox("Cap number of pairs", value=False, key="inline_use_max_pairs")
                if use_cap:
                    st.number_input("Max pairs:", min_value=1, max_value=50, value=2, step=1, key="inline_max_pairs")
            else:
                if all_cols and len(all_cols) >= 2:
                    n = st.number_input("Number of pairs:", min_value=1, max_value=20, value=1, step=1, key="inline_num_pairs")
                    for i in range(int(n)):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.selectbox(f"Pair {i+1} — Col 1:", [""] + all_cols, key=f"inline_pair_{i}_a1")
                        with c2:
                            st.selectbox(f"Pair {i+1} — Col 2:", [""] + all_cols, key=f"inline_pair_{i}_a2")
                else:
                    st.info("No columns available to build pairs from.")
            st.session_state["_stage45_mode"] = mode

    # ------------------------------------------------------------------
    # Stage 6 — Bias Mitigation
    # ------------------------------------------------------------------
    elif stage.key == "6_bias_mitigation":
        mit_cfg = pipeline._pipeline_ctx.get("mitigation_config") or {}
        preconfigured = list(mit_cfg.get("methods", {}).keys())
        with st.expander("Bias Mitigation — configure before running", expanded=True):
            st.caption("Select techniques to apply. Leave all unchecked to skip mitigation.")
            _options = ["Reweighting", "SMOTE", "Random Oversampling", "Random Undersampling"]
            for opt in _options:
                st.checkbox(opt, value=(opt in preconfigured), key=f"inline_mit_{opt}")


def _handle_submit(pipeline, user_text, action_choice):
    """Process user submission: navigate the pipeline and append messages."""

    action_map = {
        "Forward": NavigationAction.FORWARD,
        "Backward": NavigationAction.BACKWARD,
        "Repeat": NavigationAction.REPEAT,
    }
    action = action_map.get(action_choice, NavigationAction.FORWARD)

    # Record user message
    display_text = user_text.strip() if user_text.strip() else f"[{action_choice}]"
    st.session_state.chat_messages.append({"role": "user", "content": display_text})

    # --- Backward ---
    if action == NavigationAction.BACKWARD:
        result = pipeline.navigate(NavigationAction.BACKWARD, user_text)
        st.session_state.evaluation_results = pipeline.evaluation_results
        st.session_state.chat_messages.append(
            {"role": "system", "content": result.get("message", "Moved backward.")}
        )
        st.rerun()
        return

    # --- Forward / Repeat ---
    cur_stage = None
    if action == NavigationAction.FORWARD:
        cur_stage = pipeline.current_stage
    elif action == NavigationAction.REPEAT:
        idx = max(0, pipeline.current_stage_index - 1)
        cur_stage = pipeline.stages[idx] if idx < len(pipeline.stages) else None

    # Apply session-state overrides to the pipeline context
    if cur_stage:
        if st.session_state.confirmed_sensitive_columns:
            pipeline._pipeline_ctx["confirmed_sensitive_columns"] = (
                st.session_state.confirmed_sensitive_columns
            )
        if hasattr(st.session_state, "ml_model_type") and st.session_state.ml_model_type:
            pipeline._pipeline_ctx["ml_config"] = {
                "enabled": True, 
                "model_type": st.session_state.ml_model_type,
                "model_params": st.session_state.get("ml_model_params", {})
            }
        elif hasattr(st.session_state, "ml_config") and st.session_state.ml_config:
            pipeline._pipeline_ctx["ml_config"] = st.session_state.ml_config

        # Apply inline stage controls
        _apply_inline_stage_controls(pipeline, cur_stage)

        # Parse user text for special stage instructions (free-form override)
        _apply_user_text_overrides(pipeline, cur_stage, user_text)

    try:
        with st.spinner(
            f"Running {cur_stage.name if cur_stage else 'stage'}..."
        ):
            result = pipeline.navigate(action, user_text)
        st.session_state.evaluation_results = pipeline.evaluation_results

        # Figure out which stage just ran
        ran_stage = None
        if action == NavigationAction.FORWARD:
            idx = pipeline.current_stage_index - 1
            ran_stage = (
                pipeline.stages[idx] if 0 <= idx < len(pipeline.stages) else None
            )
        elif action == NavigationAction.REPEAT:
            idx = max(0, pipeline.current_stage_index - 1)
            ran_stage = (
                pipeline.stages[idx] if idx < len(pipeline.stages) else None
            )

        if ran_stage and ran_stage.is_completed:
            st.session_state.chat_messages.append(
                {
                    "role": "stage_result",
                    "content": f"**{ran_stage.name}** completed.",
                    "stage_key": ran_stage.key,
                    "stage_data": ran_stage.result,
                }
            )

            # If the next stage requires confirmation, prompt the user
            next_stage = pipeline.current_stage
            if next_stage and next_stage.requires_confirmation:
                hint = _get_confirmation_hint(next_stage, pipeline)
                if hint:
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": hint}
                    )
        elif result.get("status") == "finished":
            st.session_state.chat_messages.append(
                {"role": "system", "content": "All stages have been completed!"}
            )
        else:
            msg = result.get("message", "Stage completed.")
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": msg}
            )

    except Exception as e:
        st.session_state.chat_messages.append(
            {
                "role": "system",
                "content": f"Error: {str(e)}\n```\n{traceback.format_exc()}\n```",
            }
        )

    st.rerun()


# ======================================================================
# Inline stage control application
# ======================================================================

def _apply_inline_stage_controls(pipeline, stage):
    """Read inline widget values and apply them to pipeline context."""
    ss = st.session_state

    # Stage 3 — sensitive attribute detection
    if stage.key == "3_sensitive":
        mode = ss.get("_stage3_mode", "auto")
        if mode == "manual":
            cols = ss.get("inline_sens_cols", [])
            if cols:
                pipeline._pipeline_ctx["confirmed_sensitive_columns"] = cols
                # Also store so the post-detection filter step uses them
                ss.confirmed_sensitive_columns = cols

    # Stage 3.5 — discretization
    elif stage.key == "3_5_discretization":
        # Apply the user's attribute selection for discretization
        disc_attrs = ss.get("_stage35_attrs")
        if disc_attrs is not None:
            # Use a dedicated key so we don't collide with Stage 3's bypass key
            pipeline._pipeline_ctx["discretization_sensitive_columns"] = disc_attrs

        method = ss.get("_stage35_method", "auto")
        pipeline._pipeline_ctx["discretization_method"] = method
        if method in ("equal_width", "equal_frequency"):
            n_bins = int(ss.get("inline_disc_bins", 5))
            pipeline._pipeline_ctx["discretization_bins"] = n_bins

    # Stage 4.5 — pair evaluation
    elif stage.key == "4_5_target_fairness":
        mode = ss.get("_stage45_mode", "auto")
        if mode == "auto":
            # Always reset selected_pairs so pair selection re-runs with the
            # latest cap (pair selection happened after stage 3 before the user
            # had a chance to configure this control).
            pipeline._pipeline_ctx["selected_pairs"] = None
            if ss.get("inline_use_max_pairs"):
                max_p = int(ss.get("inline_max_pairs", 2))
                pipeline._pipeline_ctx["max_pairs"] = max_p
                pipeline._handle_pair_selection(None, max_p)
            else:
                pipeline._pipeline_ctx["max_pairs"] = None
                pipeline._handle_pair_selection(None, None)
        else:
            n = int(ss.get("inline_num_pairs", 1))
            pairs = []
            for i in range(n):
                a1 = ss.get(f"inline_pair_{i}_a1", "")
                a2 = ss.get(f"inline_pair_{i}_a2", "")
                if a1 and a2 and a1 != a2:
                    pairs.append((a1, a2))
            if pairs:
                pipeline._pipeline_ctx["selected_pairs"] = pairs
                pipeline._pipeline_ctx["user_specified_pairs"] = pairs

    # Stage 6 — bias mitigation
    elif stage.key == "6_bias_mitigation":
        _options = ["Reweighting", "SMOTE", "Random Oversampling", "Random Undersampling"]
        chosen = {opt: {} for opt in _options if ss.get(f"inline_mit_{opt}", False)}
        if chosen:
            pipeline._pipeline_ctx["mitigation_config"] = {"methods": chosen}
        else:
            # If nothing checked AND nothing was pre-configured, leave as-is (skip)
            pipeline._pipeline_ctx.setdefault("mitigation_config", None)


# ======================================================================
# User-text parsing helpers
# ======================================================================

def _apply_user_text_overrides(pipeline, stage, user_text):
    """Parse free-form user text and set pipeline_ctx overrides accordingly."""
    if not user_text or not user_text.strip():
        return

    text = user_text.strip()

    # Stage 4 -- user can list sensitive columns
    if stage.key == "4_imbalance":
        cols = [c.strip() for c in text.split(",") if c.strip()]
        if cols and not text.lower().startswith("["):
            pipeline.pipeline_ctx["confirmed_sensitive_columns"] = cols
            st.session_state.confirmed_sensitive_columns = cols

    # Stage 4.5 -- user can specify pairs like "Sex+Race, Age+Education"
    # Or type "none"/"skip" to skip the intersectional analysis
    if stage.key == "4_5_target_fairness":
        text_lower = text.strip().lower()
        if text_lower in ("none", "skip", "no", "n"):
            # User explicitly wants to skip pair analysis
            pipeline.pipeline_ctx["selected_pairs"] = []
        else:
            pairs = []
            for part in text.split(","):
                part = part.strip()
                if "+" in part:
                    tokens = [t.strip() for t in part.split("+")]
                    if len(tokens) == 2:
                        pairs.append(tuple(tokens))
            if pairs:
                pipeline.pipeline_ctx["selected_pairs"] = pairs

    # Stage 6 -- user names mitigation methods
    if stage.key == "6_bias_mitigation":
        valid_methods = {
            "reweighting": "Reweighting",
            "smote": "SMOTE",
            "random oversampling": "Random Oversampling",
            "oversampling": "Random Oversampling",
            "random undersampling": "Random Undersampling",
            "undersampling": "Random Undersampling",
        }
        chosen = {}
        for part in text.split(","):
            part_lower = part.strip().lower()
            if part_lower in valid_methods:
                method_name = valid_methods[part_lower]
                chosen[method_name] = {}
        if chosen:
            pipeline.pipeline_ctx["mitigation_config"] = {"methods": chosen}


# ======================================================================
# Confirmation hints for interactive stages
# ======================================================================

def _get_confirmation_hint(stage, pipeline):
    """Return a helpful prompt shown after the previous stage completes."""

    if stage.key == "3_sensitive":
        return (
            "**Next: Sensitive Attribute Detection**\n\n"
            "Use the panel above to choose the detection mode:\n"
            "- **Auto**: the AI will detect sensitive columns — "
            "you can review and filter the identified list afterwards.\n"
            "- **Manual**: select your own sensitive attributes from the dataset "
            "columns (AI detection is skipped).\n\n"
            "Then press **Forward**."
        )

    if stage.key == "3_5_discretization":
        sens = (
            pipeline.evaluation_results.get("stages", {})
            .get("3_sensitive", {})
            .get("sensitive_columns", [])
        )
        return (
            f"**Next: Sensitive Attribute Discretization**\n\n"
            f"Sensitive columns detected: {', '.join(sens) if sens else '—'}\n\n"
            "If any of these columns are continuous (e.g. Age), they will be discretized "
            "into bins for fairness metric computation.\n\n"
            "Use the panel above to:\n"
            "1. **Choose which attributes** to include in discretization (all are pre-selected).\n"
            "2. **Choose a discretization method** "
            "(*auto*, *equal_width*, or *equal_frequency*), then press **Forward**."
        )

    if stage.key == "4_5_target_fairness":
        sens = (
            pipeline.evaluation_results.get("stages", {})
            .get("3_sensitive", {})
            .get("sensitive_columns", [])
        )
        pairs_str = ", ".join(f"{a}+{b}" for a, b in combinations(sens, 2)) if len(sens) >= 2 else "—"
        return (
            f"**Next: Target Fairness Analysis**\n\n"
            f"Detected sensitive columns: {', '.join(sens) if sens else '—'}\n\n"
            f"Possible pairs: {pairs_str}\n\n"
            "Use the panel above to configure pair selection (auto with optional cap, "
            "or restricted to specific pairs), then press **Forward**."
        )

    if stage.key == "6_bias_mitigation":
        return (
            "**Next: Bias Mitigation**\n\n"
            "Use the panel above to select which mitigation techniques to apply "
            "(Reweighting, SMOTE, oversampling / undersampling). "
            "Leave all unchecked to skip mitigation entirely, then press **Forward**."
        )

    return ""


# ======================================================================
# Helpers
# ======================================================================

def _reset_state():
    """Reset all pipeline-related session state."""
    st.session_state.mode = None
    st.session_state.pipeline = None
    st.session_state.pipeline_started = False
    st.session_state.confirmed_sensitive_columns = None
    st.session_state.ml_config = {"enabled": False}
    st.session_state.ml_model_type = "Random Forest"
    st.session_state.ml_model_params = {}


def _stop_and_generate_report():
    """Stop the pipeline and generate a partial report with completed stages."""
    pipeline = st.session_state.pipeline
    if not pipeline:
        st.warning("No pipeline to stop.")
        return

    try:
        with st.spinner("Generating partial report..."):
            pipeline.evaluation_results["partial_report"] = True
            pipeline.generate_report()
            st.session_state.evaluation_results = pipeline.evaluation_results
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "⏹️ **Pipeline stopped.** Partial report generated with completed stages.\n\n"
                    f"**Report Directory:** `{pipeline.report_dir}`\n\n"
                    "You can download the PDF below or reset to start a new evaluation."
                ),
            })
        st.rerun()
    except Exception as e:
        st.error(f"Error generating report: {e}")
