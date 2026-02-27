"""
New Evaluation page -- chatbot-style interactive pipeline.

The pipeline is presented as a conversation between the user and the system.
Below the chat area there is a text-input field, an action selector
(Forward / Backward / Repeat) and a Send button.
"""

import os
import traceback
from datetime import datetime
from itertools import combinations

import streamlit as st

from pipeline import DatasetEvaluationPipeline
from stage import NavigationAction, StageStatus

from gui.config import (
    get_config_path,
    get_available_models,
    get_default_model_name,
    validate_api_keys,
)
from gui.utils import (
    BASE_DIR,
    get_available_datasets,
    upload_dataset,
    get_dataset_columns,
)
from gui.widgets.stage_display import display_stage_results


# ======================================================================
# Public entry-point
# ======================================================================

def new_evaluation_page():
    """Render the full *New Evaluation* page (sidebar + chatbot area)."""

    st.markdown("<div class='main-header'>New Evaluation</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Sidebar -- configuration
    # ------------------------------------------------------------------
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

        selected_dataset = st.selectbox("Select dataset", datasets)
        st.session_state.dataset_name = selected_dataset

        # ---- Model selection ----
        st.markdown("#### Model Selection")
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

        # ---- Start / Reset buttons ----
        if selected_dataset:
            if not st.session_state.pipeline_started:
                if st.button("Start Evaluation", width="stretch", type="primary"):
                    _initialize_pipeline()
            else:
                if st.button("Reset Pipeline", width="stretch"):
                    _reset_state()
                    st.rerun()

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

    # ------------------------------------------------------------------
    # Main area -- Chat interface
    # ------------------------------------------------------------------
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


# ======================================================================
# Pipeline initialisation
# ======================================================================

def _initialize_pipeline():
    """Create the pipeline, build stages, and seed the first chat message."""
    try:
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
                default_model=st.session_state.model_choice,
            )
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


# ======================================================================
# Chat interface
# ======================================================================

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
    if pipeline.current_stage_index >= len(pipeline.stages):
        st.success("All stages completed!")
        results = st.session_state.evaluation_results
        if results and "final_reports_generated" not in results:
            with st.spinner("Generating final reports..."):
                try:
                    pipeline.evaluation_results = results
                    pipeline.generate_report()
                    results["final_reports_generated"] = True
                except Exception as e:
                    st.error(f"Report generation error: {e}")
        if results and "report_directory" in results:
            st.markdown(f"**Report Directory:** `{results['report_directory']}`")

    # ---- Input area ----
    st.markdown("---")
    _render_input_area(pipeline)


def _render_input_area(pipeline):
    """Text input + action selector + send button."""

    cur = pipeline.current_stage
    if cur and not pipeline.is_finished:
        st.caption(f"Next stage: **{cur.name}** -- {cur.description}")
    elif pipeline.is_finished:
        st.caption(
            "Pipeline finished. You can **Repeat** or **Backward** to revisit stages."
        )

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
                options=["Forward", "Backward", "Repeat"],
                index=0,
                label_visibility="collapsed",
            )

        with col_send:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted:
        _handle_submit(pipeline, user_text, action_choice)


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
            pipeline.pipeline_ctx["confirmed_sensitive_columns"] = (
                st.session_state.confirmed_sensitive_columns
            )
        if st.session_state.proxy_config:
            pipeline.pipeline_ctx["proxy_config"] = st.session_state.proxy_config

        # Parse user text for special stage instructions
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
    if stage.key == "4_5_target_fairness":
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
    """Return a helpful prompt for stages that need user confirmation."""

    if stage.key == "4_imbalance":
        sens = (
            pipeline.evaluation_results.get("stages", {})
            .get("3_sensitive", {})
            .get("sensitive_columns", [])
        )
        if sens:
            return (
                f"**Stage 3 detected these sensitive columns:** {', '.join(sens)}\n\n"
                "Before proceeding with the imbalance analysis you can:\n"
                "- Type column names to override (comma-separated)\n"
                "- Or just press **Forward** to accept and continue."
            )

    if stage.key == "4_5_target_fairness":
        sens = (
            pipeline.evaluation_results.get("stages", {})
            .get("3_sensitive", {})
            .get("sensitive_columns", [])
        )
        if len(sens) >= 2:
            pairs = list(combinations(sens, 2))
            pairs_str = ", ".join(f"{a}+{b}" for a, b in pairs)
            return (
                f"**Target Fairness Analysis** can examine these attribute pairs:\n"
                f"{pairs_str}\n\n"
                "Type the pairs you want (e.g. `Sex+Race, Age+Education`) "
                "or press **Forward** to analyse all."
            )

    if stage.key == "6_bias_mitigation":
        return (
            "**Bias Mitigation** -- choose which techniques to apply.\n\n"
            "Options: `Reweighting`, `SMOTE`, `Random Oversampling`, "
            "`Random Undersampling`\n\n"
            "Type your choices (comma-separated) or press **Forward** to skip."
        )

    return ""


# ======================================================================
# Helpers
# ======================================================================

def _reset_state():
    """Reset all pipeline-related session state."""
    st.session_state.mode = None
    st.session_state.current_step = 0
    st.session_state.pipeline = None
    st.session_state.pipeline_started = False
    st.session_state.evaluation_results = None
    st.session_state.chat_messages = []
    st.session_state.confirmed_sensitive_columns = None
    st.session_state.proxy_config = {"enabled": False}
