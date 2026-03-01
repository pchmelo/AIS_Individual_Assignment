import os
import re
import streamlit as st
import pandas as pd

from gui.widgets.results import (
    display_quality_results,
    display_sensitive_results,
    display_imbalance_results,
    display_fairness_results,
)
from gui.widgets.fairness import render_fairness_comparison_board


def _clean_agent_analysis(text: str) -> str:
    """Remove tool_call artifacts and other LLM formatting noise from agent output."""
    if not text:
        return ""
    # Remove <tool_call>...</tool_call> blocks (including malformed ones)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # Remove standalone <tool_call> or </tool_call> tags
    text = re.sub(r'</?tool_call>', '', text)
    # Remove <function=...>...</function> blocks
    text = re.sub(r'<function=[^>]*>.*?</function>', '', text, flags=re.DOTALL)
    # Remove <parameter=...>...</parameter> blocks
    text = re.sub(r'<parameter=[^>]*>.*?</parameter>', '', text, flags=re.DOTALL)
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage result renderer
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    "0_loading": "Stage 0: Dataset Loading",
    "1_objective": "Stage 1: Objective Inspection",
    "2_quality": "Stage 2: Data Quality Analysis",
    "3_sensitive": "Stage 3: Sensitive Attribute Detection",
    "4_imbalance": "Stage 4: Imbalance Analysis",
    "4_5_target_fairness": "Stage 4.5: Target Fairness Analysis",
    "5_recommendations": "Stage 5: Recommendations",
    "6_bias_mitigation": "Stage 6: Bias Mitigation",
}


def display_stage_results(stage_key: str, stage_result: dict):
    """Render the results of a completed stage inside the stepwise UI."""

    st.markdown(f"### {STAGE_NAMES.get(stage_key, stage_key)}")

    # ------------------------------------------------------------------
    # Bias mitigation (stage 6)
    # ------------------------------------------------------------------
    if stage_key == "6_bias_mitigation":
        _render_bias_mitigation(stage_result)
        return

    # ------------------------------------------------------------------
    # Generic tool_result rendering
    # ------------------------------------------------------------------
    if "tool_result" in stage_result:
        tool_result = stage_result["tool_result"]

        if stage_key == "0_loading":
            if tool_result.get("status") == "success":
                st.success(
                    f"Dataset loaded: {tool_result.get('rows', 0)} rows, "
                    f"{len(tool_result.get('columns', []))} columns"
                )
                with st.expander("View Columns"):
                    cols = tool_result.get("columns", [])
                    st.write(", ".join(f"`{c}`" for c in cols))
            else:
                st.error(f"Failed to load dataset: {tool_result.get('error', 'Unknown error')}")

        elif stage_key == "2_quality":
            display_quality_results(tool_result)

        elif stage_key == "3_sensitive":
            display_sensitive_results(stage_result)

        elif stage_key == "4_imbalance":
            display_imbalance_results(stage_result)

        elif stage_key == "4_5_target_fairness":
            display_fairness_results(stage_result)

        elif isinstance(tool_result, dict) and tool_result:
            with st.expander("Tool Result"):
                st.json(tool_result)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    if stage_key == "5_recommendations" and "recommendations" in stage_result:
        with st.expander("Recommendations", expanded=True):
            st.markdown(stage_result["recommendations"])

    # ------------------------------------------------------------------
    # Agent analysis
    # ------------------------------------------------------------------
    if "agent_analysis" in stage_result:
        analysis_text = _clean_agent_analysis(stage_result["agent_analysis"])

        if analysis_text and str(analysis_text).strip():
            if stage_key == "3_sensitive":
                _render_sensitive_analysis(stage_result, analysis_text)
            else:
                st.markdown("---")
                stage_label = STAGE_NAMES.get(stage_key, stage_key).split(":")[0]
                with st.expander(f"Agent Analysis - {stage_label}", expanded=True):
                    st.markdown(analysis_text)

    # ------------------------------------------------------------------
    # Objective details
    # ------------------------------------------------------------------
    if stage_key == "1_objective" and "objective" in stage_result:
        st.info(f"**Objective:** {stage_result['objective']}")
        st.write(f"**Audit Request:** {'Yes' if stage_result.get('is_audit_request') else 'No'}")
        st.write(f"**Validation:** {stage_result.get('validation', 'N/A')}")


# ======================================================================
# Private helpers
# ======================================================================

def _render_sensitive_analysis(stage_result: dict, analysis_text: str):
    """Render the agent-analysis section of the sensitive-detection stage."""

    # Column summary table
    if "simplified_summary" in stage_result:
        with st.expander("Column Summary Table"):
            summary_text = stage_result["simplified_summary"]
            lines = summary_text.strip().split("\n")

            data_lines = []
            for line in lines:
                if line and not line.startswith("=") and not line.startswith("COLUMN") and not ("Column" in line and "Type" in line):
                    if not all(c in "= \t" for c in line):
                        data_lines.append(line)

            table_data = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 4:
                    table_data.append({
                        "Column": parts[0],
                        "Type": parts[1],
                        "Unique": parts[2],
                        "Sample Values": " ".join(parts[3:]),
                    })

            if table_data:
                try:
                    df_summary = pd.DataFrame(table_data)
                    st.dataframe(df_summary, width="stretch", hide_index=True)
                except Exception:
                    st.text(summary_text)
            else:
                st.text(summary_text)

    # Structured sensitive-column table
    pattern = r"Column:\s*([^\|]+)\s*\|\s*Reason:\s*([^\|]+)\s*\|\s*Values:\s*(.+?)(?=Column:|$)"
    matches = re.findall(pattern, analysis_text, re.DOTALL)

    if matches:
        st.markdown("---")
        st.markdown("**Identified Sensitive Attributes:**")

        table_data = []
        for col, reason, values in matches:
            table_data.append({
                "Column": col.strip(),
                "Reason": reason.strip(),
                "Values": values.strip().replace("\n", " "),
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, width="stretch", hide_index=True)
        st.info(f"Total: {len(table_data)} sensitive attributes identified")


# ------------------------------------------------------------------
# Bias mitigation (stage 6) rendering
# ------------------------------------------------------------------

def _render_bias_mitigation(stage_result: dict):
    """Full renderer for the bias-mitigation stage."""

    if stage_result.get("status") == "skipped":
        st.info("Bias mitigation was skipped by user.")
        return
    elif stage_result.get("status") == "error":
        st.error(
            f"Error applying {stage_result.get('method', 'mitigation')}: "
            f"{stage_result.get('error', 'Unknown error')}"
        )
        return
    elif stage_result.get("status") != "success":
        return

    if "methods" in stage_result:
        _render_multi_method_mitigation(stage_result)
    else:
        _render_single_method_mitigation(stage_result)


def _render_multi_method_mitigation(stage_result: dict):
    """Render the dashboard for multiple mitigation methods."""
    methods_results = stage_result["methods"]
    applied_methods = stage_result.get("applied_methods", list(methods_results.keys()))

    st.success(f"Successfully applied {len(applied_methods)} method(s)!")

    st.markdown("---")
    st.markdown("### Methods Comparison Dashboard")

    # Summary comparison table
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
                "Improvement": "\u2713" if improved == "Yes" else "\u2717",
            })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width="stretch", hide_index=True)

        best_method = None
        best_ratio = float("inf")
        for method in applied_methods:
            mr = methods_results.get(method, {})
            if mr.get("status") == "success":
                imb = mr.get("comparison_result", {}).get("imbalance_metrics", {})
                rat = imb.get("mitigated_imbalance_ratio", float("inf"))
                if rat < best_ratio:
                    best_ratio = rat
                    best_method = method
        if best_method:
            st.info(f"**Best Method:** {best_method} achieved the lowest imbalance ratio ({best_ratio:.2f})")
    else:
        st.warning("No successful methods to compare.")

    st.markdown("---")
    st.markdown("### Individual Method Results")

    for method in applied_methods:
        method_result = methods_results.get(method, {})

        with st.expander(f"{method} - Detailed Results", expanded=False):
            if method_result.get("status") == "error":
                st.error(f"Error: {method_result.get('error', 'Unknown error')}")
                continue

            _render_method_detail(method, method_result)


def _render_single_method_mitigation(stage_result: dict):
    st.success(f"{stage_result.get('method', 'Bias mitigation')} applied successfully!")

    mitigation_result = stage_result.get("mitigation_result", {})
    comparison_result = stage_result.get("comparison_result", {})

    _render_mitigation_summary(mitigation_result)
    _render_distribution_comparison(mitigation_result)
    _render_imbalance_improvement(comparison_result)

    if comparison_result.get("agent_analysis"):
        st.markdown("---")
        st.markdown("#### Agent Analysis")
        with st.expander("View Detailed Analysis", expanded=True):
            st.markdown(_clean_agent_analysis(comparison_result["agent_analysis"]))

    _render_download_button(mitigation_result, key_prefix="download_mitigated")


def _render_method_detail(method: str, method_result: dict):
    """Detail view inside an expander for one mitigation method."""
    mitigation_result = method_result.get("mitigation_result", {})
    comparison_result = method_result.get("comparison_result", {})

    st.markdown("#### Summary")
    _render_mitigation_summary(mitigation_result)

    st.markdown("---")
    st.markdown("#### Target Distribution Comparison")
    _render_distribution_comparison(mitigation_result)

    _render_imbalance_improvement(comparison_result)

    if method_result.get("fairness_comparison"):
        st.markdown("---")
        render_fairness_comparison_board(
            comparison_data=method_result["fairness_comparison"],
            method_name=method,
        )

    if comparison_result.get("agent_analysis"):
        st.markdown("---")
        st.markdown("#### Agent Analysis")
        st.markdown(_clean_agent_analysis(comparison_result["agent_analysis"]))

    _render_download_button(mitigation_result, key_prefix=f"download_{method.replace(' ', '_')}")


# ------------------------------------------------------------------
# Shared sub-renderers
# ------------------------------------------------------------------

def _render_mitigation_summary(mitigation_result: dict):
    col1, col2, col3, col4 = st.columns(4)
    original_rows = mitigation_result.get("original_rows", 0)

    with col1:
        st.metric("Original Rows", f"{original_rows:,}")
    with col2:
        new_rows = mitigation_result.get("new_rows", original_rows)
        st.metric("New Rows", f"{new_rows:,}")
    with col3:
        if "rows_added" in mitigation_result:
            st.metric("Rows Added", f"+{mitigation_result['rows_added']:,}", delta=mitigation_result["rows_added"])
        elif "rows_removed" in mitigation_result:
            st.metric("Rows Removed", f"-{mitigation_result['rows_removed']:,}", delta=-mitigation_result["rows_removed"])
    with col4:
        output_file = mitigation_result.get("output_file", "")
        if output_file:
            filename = os.path.basename(output_file)
            st.metric("Output File", "\u2713")
            st.caption(filename)


def _render_distribution_comparison(mitigation_result: dict):
    dist_before = mitigation_result.get("distribution_before", {})
    dist_after = mitigation_result.get("distribution_after", dist_before)

    if not dist_before or not dist_after:
        return

    all_values = set(dist_before.keys()) | set(dist_after.keys())
    rows = []
    for value in sorted(all_values):
        before_count = dist_before.get(value, 0)
        after_count = dist_after.get(value, 0)
        before_pct = (before_count / sum(dist_before.values()) * 100) if dist_before else 0
        after_pct = (after_count / sum(dist_after.values()) * 100) if dist_after else 0

        rows.append({
            "Class": str(value),
            "Before Count": before_count,
            "Before %": f"{before_pct:.2f}%",
            "After Count": after_count,
            "After %": f"{after_pct:.2f}%",
            "Change": after_count - before_count,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def _render_imbalance_improvement(comparison_result: dict):
    if not comparison_result.get("imbalance_metrics"):
        return

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
        st.metric("Mitigated Imbalance Ratio", f"{mit_ratio:.2f}", delta=f"{delta:.2f}", delta_color="inverse")
    with col3:
        improved = imb_metrics.get("improvement", "No")
        if improved == "Yes":
            st.success("Imbalance Improved")
        else:
            st.warning("No Improvement")


def _render_download_button(mitigation_result: dict, key_prefix: str = "download"):
    output_file = mitigation_result.get("output_file", "")
    if output_file and os.path.exists(output_file):
        st.markdown("---")
        # Use hash of output file path to ensure unique key
        file_hash = hash(output_file) % 10**8
        unique_key = f"{key_prefix}_{file_hash}"
        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Mitigated Dataset",
                data=f,
                file_name=os.path.basename(output_file),
                mime="text/csv",
                key=unique_key,
            )
