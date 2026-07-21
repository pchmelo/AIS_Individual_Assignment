import json
import os
import re
import traceback

import pandas as pd
import streamlit as st

from gui.utils import BASE_DIR, get_available_reports, parse_report_file
from gui.widgets.fairness import render_fairness_board, render_fairness_comparison_board
from gui.widgets.stage_display import _clean_agent_analysis
from gui.pdf_generator import generate_pdf_bytes


def view_results_page():
    st.markdown("<div class='main-header'>Previous Results</div>", unsafe_allow_html=True)

    if st.button("\u2190 Back to Main"):
        st.session_state.mode = None
        st.session_state.selected_report = None
        st.rerun()

    reports = get_available_reports()

    if not reports:
        st.warning("No previous reports found.")
        return

    selected_report = st.selectbox("Select a report to view:", reports)

    if not selected_report:
        return

    report_dir = os.path.join(BASE_DIR, "reports", selected_report)
    report_file = os.path.join(report_dir, "evaluation_report.md")
    json_data_file = os.path.join(report_dir, "stage_data.json")

    # PDF Download button - use evaluation_report.md for markdown formatting
    if os.path.exists(report_file):
        col1, col2 = st.columns([3, 1])
        with col2:
            try:
                pdf_bytes = generate_pdf_bytes(report_file)
                dataset_name = selected_report.split("_")[0]
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"{dataset_name}_fairness_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_view_results",
                )
            except Exception as e:
                st.warning(f"PDF: {e}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Full Report", "Stage Data", "Recommendations", "Bias Mitigation", "Visualizations"]
    )

    with tab1:
        _display_parsed_report(report_file, "Full Report")

    with tab2:
        _display_json_data(json_data_file, "Stage Data")

    with tab3:
        _render_recommendations_tab(report_file)

    with tab4:
        _render_bias_mitigation_tab(report_dir, report_file)

    with tab5:
        _render_visualizations_tab(report_dir)


# ======================================================================
# Parsed report renderer
# ======================================================================

def _display_parsed_report(filepath: str, report_type: str = "Full Report"):
    result = parse_report_file(filepath)

    if not result:
        st.warning(f"{report_type} file not found")
        return

    header_info, stages = result

    if header_info:
        with st.container():
            st.markdown("### Report Information")
            cols = st.columns(3)
            if "dataset" in header_info:
                cols[0].metric("Dataset", header_info["dataset"])
            if "timestamp" in header_info:
                cols[1].metric("Timestamp", header_info["timestamp"])
            if "target" in header_info:
                cols[2].metric("Target Column", header_info["target"])

            if "objective" in header_info:
                st.info(f"**Objective:** {header_info['objective']}")

        st.markdown("---")

    if stages:
        st.markdown("### Report Stages")
        for stage_name, stage_content in stages.items():
            with st.expander(stage_name, expanded=False):
                _render_stage_content(stage_content)
    else:
        st.warning("No stages found in report")


def _display_json_data(filepath: str, report_type: str = "Stage Data"):
    """Display stage data from JSON file with structured view."""
    if not os.path.exists(filepath):
        st.warning(f"{report_type} file not found")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return

    # Display metadata
    if "metadata" in data:
        meta = data["metadata"]
        with st.container():
            st.markdown("### Evaluation Metadata")
            cols = st.columns(3)
            if "dataset" in meta:
                cols[0].metric("Dataset", meta["dataset"])
            if "timestamp" in meta:
                cols[1].metric("Timestamp", meta["timestamp"])
            if "target_column" in meta:
                cols[2].metric("Target Column", meta["target_column"])
        st.markdown("---")

    # Display stages
    if "stages" in data:
        st.markdown("### Stage Tool Results")
        for stage_name, stage_data in data["stages"].items():
            with st.expander(stage_name, expanded=False):
                if isinstance(stage_data, dict):
                    if "tool_used" in stage_data:
                        st.markdown(f"**Tool:** `{stage_data['tool_used']}`")
                    if "tool_result" in stage_data:
                        result = stage_data["tool_result"]
                        if isinstance(result, dict):
                            st.json(result)
                        else:
                            st.write(result)
                else:
                    st.json(stage_data)
    else:
        st.warning("No stage data found in JSON file")


def _render_stage_content(stage_content: str):
    """Parse and display a single stage's content (including markers)."""

    has_markers = (
        "[TOOL USED]" in stage_content
        or "[AGENT ANALYSIS]" in stage_content
        or "[ML MODEL RESULTS]" in stage_content
    )

    if has_markers:
        sections = stage_content.split("[TOOL USED]")

        for section in sections:
            if not section.strip():
                continue

            # ---- Debug section ----
            with st.expander("Debug Section Parsing"):
                st.write(f"Has [MITIGATION RESULTS]: {'[MITIGATION RESULTS]' in section}")
                if "[MITIGATION RESULTS]" in section:
                    mit_json_dbg = _extract_json_local(section, "[MITIGATION RESULTS]")
                    st.write(f"Mitigation JSON extracted: {bool(mit_json_dbg)}")
                    if not mit_json_dbg:
                        st.write("Mitigation extraction failed.")

                st.write(f"Has [COMPARISON RESULTS]: {'[COMPARISON RESULTS]' in section}")
                if "[COMPARISON RESULTS]" in section:
                    comp_json_dbg = _extract_json_local(section, "[COMPARISON RESULTS]")
                    st.write(f"Comparison JSON extracted: {bool(comp_json_dbg)}")

                st.write(f"Has [FAIRNESS COMPARISON]: {'[FAIRNESS COMPARISON]' in section}")
                if "[FAIRNESS COMPARISON]" in section:
                    fair_json_dbg = _extract_json_local(section, "[FAIRNESS COMPARISON]")
                    st.write(f"Fairness JSON extracted: {bool(fair_json_dbg)}")

            # ---- ML Model / Intersectional / Mitigation / Comparison / Fairness ----
            if "[ML MODEL RESULTS]" in section or "[INTERSECTIONAL ML RESULTS]" in section:

                if "[ML MODEL RESULTS]" in section:
                    ml_json = _extract_json_local(section, "[ML MODEL RESULTS]")
                    if ml_json:
                        try:
                            ml_data = json.loads(ml_json)
                            if ml_data and isinstance(ml_data, dict) and ml_data.get("status") == "success":
                                st.markdown("---")
                                render_fairness_board(ml_data, title="ML Model Fairness Analysis (Stage 4)")
                        except json.JSONDecodeError as e:
                            st.warning(f"Could not parse ML model results (JSON error at position {e.pos})")
                            with st.expander("View Raw Data (for debugging)"):
                                st.text(f"Error: {str(e)}")
                                st.text(f"First 500 chars: {ml_json[:500]}")
                        except Exception as e:
                            st.warning(f"Error displaying ML model results: {type(e).__name__}: {str(e)}")

                if "[INTERSECTIONAL ML RESULTS]" in section:
                    intersect_json = _extract_json_local(section, "[INTERSECTIONAL ML RESULTS]")
                    if intersect_json:
                        try:
                            intersect_data = json.loads(intersect_json)
                            if intersect_data and isinstance(intersect_data, dict) and intersect_data.get("status") == "success":
                                st.markdown("---")
                                render_fairness_board(intersect_data, title="Intersectional Fairness Analysis (Stage 4.5)")
                        except json.JSONDecodeError as e:
                            st.warning(f"Could not parse intersectional ML results (JSON error at position {e.pos})")
                            with st.expander("View Raw Data (for debugging)"):
                                st.text(f"Error: {str(e)}")
                                st.text(f"First 500 chars: {intersect_json[:500]}")
                        except Exception as e:
                            st.warning(f"Error displaying intersectional results: {type(e).__name__}: {str(e)}")

                if "[MITIGATION RESULTS]" in section:
                    mit_json = _extract_json_local(section, "[MITIGATION RESULTS]")
                    if mit_json:
                        try:
                            mit_data = json.loads(mit_json)
                            if mit_data and isinstance(mit_data, dict):
                                st.markdown("### Mitigation Results")
                                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                                with m_col1:
                                    st.metric("Method", mit_data.get("method", "Unknown"))
                                with m_col2:
                                    st.metric("Original Rows", f"{mit_data.get('original_rows', 0):,}")
                                with m_col3:
                                    st.metric("New Rows", f"{mit_data.get('new_rows', 0):,}")
                                with m_col4:
                                    st.metric("Rows Added", f"{mit_data.get('rows_added', 0):,}")
                                st.divider()
                        except Exception as e:
                            st.warning(f"Error displaying mitigation results: {str(e)}")

                if "[COMPARISON RESULTS]" in section:
                    comp_json = _extract_json_local(section, "[COMPARISON RESULTS]")
                    if comp_json:
                        try:
                            comp_data = json.loads(comp_json)
                            if comp_data and isinstance(comp_data, dict):
                                _render_comparison_results(comp_data)
                        except Exception as e:
                            st.warning(f"Error displaying mitigation comparison: {str(e)}")

                if "[FAIRNESS COMPARISON]" in section:
                    comparison_json = _extract_json_local(section, "[FAIRNESS COMPARISON]")
                    if comparison_json:
                        try:
                            comparison_data = json.loads(comparison_json)
                            if comparison_data and isinstance(comparison_data, dict):
                                if comparison_data.get("per_attribute_comparison"):
                                    method_name = comparison_data.get("method", "Mitigation Method")
                                    st.markdown("---")
                                    render_fairness_comparison_board(
                                        comparison_data=comparison_data, method_name=method_name,
                                    )
                                else:
                                    st.info(
                                        f"Fairness comparison data incomplete for "
                                        f"{comparison_data.get('method', 'this method')} "
                                        "- no per-attribute metrics found"
                                    )
                        except json.JSONDecodeError as e:
                            st.warning(f"Could not parse fairness comparison (JSON error at position {e.pos})")
                            with st.expander("View Raw Data (for debugging)"):
                                st.text(f"Error: {str(e)}")
                                st.text(f"First 500 chars: {comparison_json[:500]}")
                        except Exception as e:
                            st.warning(f"Error displaying fairness comparison: {type(e).__name__}: {str(e)}")
                            with st.expander("View Error Details (for debugging)"):
                                st.code(traceback.format_exc())

            # Tool + agent analysis
            if "[TOOL RESULT]" in section and "[AGENT ANALYSIS]" in section:
                parts = section.split("[TOOL RESULT]")
                tool_name = parts[0].strip()
                remaining = parts[1].split("[AGENT ANALYSIS]")
                tool_result = remaining[0].strip()
                agent_analysis = remaining[1].strip() if len(remaining) > 1 else ""

                if tool_name:
                    st.markdown(f"**Tool Used:** `{tool_name}`")
                if tool_result:
                    with st.expander("Tool Result", expanded=False):
                        st.code(tool_result, language="json")
                if agent_analysis:
                    st.markdown("**Agent Analysis:**")
                    st.markdown(_clean_agent_analysis(agent_analysis))

            elif "[AGENT ANALYSIS]" in section:
                parts = section.split("[AGENT ANALYSIS]")
                st.markdown("**Agent Analysis:**")
                st.markdown(_clean_agent_analysis(parts[1].strip()))

            elif "[TOOL RESULT]" in section and "[ML MODEL RESULTS]" not in section:
                parts = section.split("[TOOL RESULT]")
                tool_name = parts[0].strip()
                if tool_name:
                    st.markdown(f"**Tool Used:** `{tool_name}`")
                if len(parts) > 1:
                    with st.expander("Tool Result", expanded=False):
                        st.code(parts[1].strip(), language="json")

    elif "[RECOMMENDATIONS]" in stage_content:
        parts = stage_content.split("[RECOMMENDATIONS]")
        st.markdown(parts[1].strip() if len(parts) > 1 else stage_content)

    else:
        st.markdown(stage_content)


# ======================================================================
# Tab helpers
# ======================================================================

def _render_recommendations_tab(report_file: str):
    st.markdown("### Stage 5: Recommendation Synthesis")
    if not os.path.exists(report_file):
        st.info("Report file not found.")
        return

    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()

    rec_start = -1
    rec_end = -1

    # Try multiple patterns for finding recommendations section
    patterns = [
        "## Stage 5: Recommendations",
        "[RECOMMENDATIONS]",
        "5_RECOMMENDATIONS",
    ]
    
    for pattern in patterns:
        if pattern in content:
            rec_start = content.find(pattern)
            break
    
    if rec_start >= 0:
        temp = content[rec_start:]
        next_markers = [
            temp.find("\n## Stage 6:"),
            temp.find("\n\n---\n\n## Stage 6"),
            temp.find("\n\n6_BIAS_MITIGATION"),
            temp.find("\n\nSTAGE 6:"),
            temp.find("\n\n================================================================================\nEND OF REPORT"),
        ]
        next_markers = [m for m in next_markers if m > 0]
        if next_markers:
            rec_end = rec_start + min(next_markers)

    if rec_start >= 0:
        rec_section = content[rec_start:rec_end] if rec_end > rec_start else content[rec_start:]
        st.markdown(rec_section)
    else:
        st.info("No recommendations found in this report.")


def _render_bias_mitigation_tab(report_dir: str, report_file: str):
    st.markdown("### Stage 6: Bias Mitigation Results")

    methods_analysis: dict[str, str] = {}

    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        if "6_BIAS_MITIGATION" in content:
            bias_start = content.find("\n\n6_BIAS_MITIGATION")
            if bias_start >= 0:
                temp = content[bias_start:]
                end_markers = [
                    temp.find("\n\n================================================================================\nEND OF REPORT"),
                ]
                end_markers = [m for m in end_markers if m > 0]

                bias_section = content[bias_start : bias_start + min(end_markers)] if end_markers else temp

                method_pattern = r"\[([A-Z][A-Z\s]+)\]\n-{40}"
                method_matches = list(re.finditer(method_pattern, bias_section))

                for i, match in enumerate(method_matches):
                    method_name = match.group(1).strip()
                    method_start = match.end()
                    method_end = method_matches[i + 1].start() if i + 1 < len(method_matches) else len(bias_section)
                    method_content = bias_section[method_start:method_end]

                    if "[AGENT ANALYSIS]" in method_content:
                        analysis_start = method_content.find("[AGENT ANALYSIS]") + len("[AGENT ANALYSIS]")
                        analysis_text = method_content[analysis_start:].strip()
                        if "\n[" in analysis_text:
                            analysis_text = analysis_text[: analysis_text.find("\n[")].strip()
                        methods_analysis[method_name] = analysis_text

    mitigation_dir = os.path.join(report_dir, "mitigation")
    generated_csv_dir = os.path.join(report_dir, "generated_csv")
    
    if os.path.exists(mitigation_dir):
        data_dir = mitigation_dir
    elif os.path.exists(generated_csv_dir):
        data_dir = generated_csv_dir
    else:
        st.info("No bias mitigation was applied in this evaluation.")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        st.info("No bias mitigation was applied in this evaluation.")
        return

    st.success(f"Found {len(csv_files)} mitigated dataset(s)")

    methods_data: dict[str, str] = {}
    for csv_file in csv_files:
        lower = csv_file.lower()
        if "smote" in lower:
            methods_data["SMOTE"] = csv_file
        elif "aif360_reweighed" in lower:
            methods_data["AIF360 Reweighing"] = csv_file
        elif "reweighted" in lower:
            methods_data["Reweighting"] = csv_file
        elif "oversampled" in lower:
            methods_data["Random Oversampling"] = csv_file
        elif "undersampled" in lower:
            methods_data["Random Undersampling"] = csv_file

    if not methods_data:
        st.info("Generated CSV files found, but could not identify mitigation methods.")
        return

    # --- Comparison table ---
    st.markdown("#### Methods Comparison")
    comparison_data = []
    for method, filename in methods_data.items():
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
            comparison_data.append({"Method": method, "Rows": f"{len(df):,}", "File": filename})
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")

    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)

    # --- Individual details ---
    st.markdown("---")
    st.markdown("#### Individual Method Details")

    for method, filename in methods_data.items():
        with st.expander(f"{method} - Detailed Results"):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)

                st.markdown("##### Dataset Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Has Weights", "Yes" if "sample_weight" in df.columns else "No")

                st.markdown("##### Column Names")
                st.write(", ".join(df.columns.tolist()))

                st.markdown("##### Sample Data (First 5 Rows)")
                st.dataframe(df.head(), width="stretch")

                # Fairness comparison
                st.markdown("---")
                st.markdown("##### Fairness Metrics Comparison")

                fairness_json_filename = f"fairness_comparison_{method.lower().replace(' ', '_')}.json"
                fairness_json_path = os.path.join(report_dir, fairness_json_filename)

                if os.path.exists(fairness_json_path):
                    try:
                        with open(fairness_json_path, "r", encoding="utf-8") as f:
                            fairness_data = json.load(f)
                        if fairness_data and isinstance(fairness_data, dict):
                            render_fairness_comparison_board(comparison_data=fairness_data, method_name=method)
                        else:
                            st.info("Fairness comparison data structure is invalid")
                    except Exception as e:
                        st.warning(f"Could not load fairness comparison: {str(e)}")
                else:
                    st.info(f"No fairness comparison available for {method} (baseline metrics may not have been generated)")

                # Agent analysis
                method_upper = method.upper()
                analysis_text = methods_analysis.get(method_upper)
                if not analysis_text and method == "Random Oversampling":
                    analysis_text = methods_analysis.get("RANDOM OVERSAMPLING")
                if not analysis_text and method == "Random Undersampling":
                    analysis_text = methods_analysis.get("RANDOM UNDERSAMPLING")

                if analysis_text:
                    st.markdown("---")
                    st.markdown("##### Agent Analysis")
                    st.markdown(analysis_text)

                # Download
                st.markdown("---")
                with open(filepath, "rb") as f:
                    st.download_button(
                        label=f"Download {method} Dataset",
                        data=f,
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_prev_{method.replace(' ', '_')}",
                    )

            except Exception as e:
                st.error(f"Error displaying {filename}: {str(e)}")


def _render_visualizations_tab(report_dir: str):
    images_dir = os.path.join(report_dir, "images")
    if not os.path.exists(images_dir):
        st.info("No images directory found")
        return

    image_files = []
    for root, _dirs, files in os.walk(images_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, file))

    if not image_files:
        st.info("No images found in this report")
        return

    st.markdown(f"**Found {len(image_files)} visualizations**")

    main_images = []
    combination_images: dict[str, list] = {}

    for img_path in image_files:
        if "_combinations" in img_path:
            path_parts = img_path.split(os.sep)
            combo_folder = None
            for part in path_parts:
                if part.endswith("_combinations"):
                    combo_folder = part.replace("_combinations", "")
                    break
            if combo_folder:
                combo_display = combo_folder.replace("_", " + ")
                combination_images.setdefault(combo_display, []).append(img_path)
        else:
            main_images.append(img_path)

    if main_images:
        st.markdown("#### Main Visualizations")
        main_image_options = {}
        for img_path in main_images:
            if os.path.exists(img_path):
                filename = os.path.basename(img_path)
                display_name = filename.replace(".png", "").replace("_", " ").title()
                main_image_options[display_name] = img_path

        if main_image_options:
            selected_main = st.selectbox(
                "Select visualization to view:",
                options=["None"] + list(main_image_options.keys()),
                key="prev_main_viz_selector",
            )
            if selected_main != "None":
                try:
                    st.image(main_image_options[selected_main], caption=selected_main, width="stretch")
                except FileNotFoundError:
                    st.warning(f"Could not read image file: {selected_main}")

    if combination_images:
        st.markdown("---")
        st.markdown("#### Combined Sensitive Attribute Analysis")

        total_charts = sum(len(imgs) for imgs in combination_images.values())
        st.info(f"{len(combination_images)} attribute combinations available ({total_charts} total charts)")

        combo_options = sorted(combination_images.keys())
        selected_combo = st.selectbox(
            "Select attribute combination to analyze:",
            options=["None"] + combo_options,
            help="Choose which combination of sensitive attributes you want to view",
            key="prev_combo_selector",
        )

        if selected_combo != "None":
            st.markdown(f"##### {selected_combo}")
            combo_imgs = combination_images[selected_combo]
            combo_image_options = {}
            for img_path in combo_imgs:
                if os.path.exists(img_path):
                    filename = os.path.basename(img_path)
                    if "scale.png" in filename:
                        display_name = filename.replace("_scale.png", "").upper() + " Scale"
                    elif "individual_combinations" in img_path:
                        display_name = filename.replace(".png", "").replace("_", " - ")
                    else:
                        display_name = filename.replace(".png", "").replace("_", " ").title()
                    combo_image_options[display_name] = img_path

            if combo_image_options:
                selected_combo_img = st.selectbox(
                    f"Select {selected_combo} visualization:",
                    options=["None"] + list(combo_image_options.keys()),
                    key=f"prev_combo_img_selector_{selected_combo.replace(' + ', '_')}",
                )
                if selected_combo_img != "None":
                    try:
                        st.image(
                            combo_image_options[selected_combo_img],
                            caption=selected_combo_img,
                            width="stretch",
                        )
                    except FileNotFoundError:
                        st.warning(f"Could not read image file: {selected_combo_img}")


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_json_local(text: str, start_marker: str):
    try:
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return None
        json_start = start_idx + len(start_marker)
        remaining = text[json_start:].strip()
        next_section = re.search(r"\n\n\[", remaining)
        if next_section:
            return remaining[: next_section.start()].strip()
        return remaining.strip()
    except Exception:
        return None


def _render_comparison_results(comp_data: dict):
    if "imbalance_metrics" in comp_data:
        st.markdown("#### Imbalance Improvement")
        imb_metrics = comp_data["imbalance_metrics"]
        col1, col2, col3 = st.columns(3)
        with col1:
            orig_ratio = float(imb_metrics.get("original_imbalance_ratio", 0))
            st.metric("Original Imbalance Ratio", f"{orig_ratio:.2f}")
        with col2:
            mit_ratio = float(imb_metrics.get("mitigated_imbalance_ratio", 0))
            delta = mit_ratio - orig_ratio
            st.metric("Mitigated Imbalance Ratio", f"{mit_ratio:.2f}", delta=f"{delta:.2f}", delta_color="inverse")
        with col3:
            improved = imb_metrics.get("improvement", "No")
            if improved == "Yes":
                st.success("Imbalance Improved")
            else:
                st.warning("No Improvement")

    if "target_distribution" in comp_data:
        st.markdown("#### Target Distribution Comparison")
        target_dist_data = comp_data["target_distribution"]

        if target_dist_data:
            rows = []
            for cls in sorted(target_dist_data.keys()):
                stats = target_dist_data[cls]
                orig_count = stats.get("original_count", 0)
                orig_pct = stats.get("original_percentage", 0)

                if "mitigated_weighted_count" in stats:
                    mit_count = stats.get("mitigated_weighted_count", 0)
                    mit_pct = stats.get("mitigated_weighted_percentage", 0)
                    change = stats.get("weighted_change", 0)
                else:
                    mit_count = stats.get("mitigated_count", 0)
                    mit_pct = stats.get("mitigated_percentage", 0)
                    change = stats.get("change", 0)

                rows.append({
                    "Class": cls,
                    "Before Count": orig_count,
                    "Before %": f"{orig_pct}%",
                    "After Count": mit_count,
                    "After %": f"{mit_pct}%",
                    "Change": change,
                })

            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
