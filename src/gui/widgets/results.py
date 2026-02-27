"""
Result Display Widgets

Reusable Streamlit components for rendering stage-level data results
(quality, sensitive columns, imbalance, fairness).
"""

import os
import streamlit as st
import pandas as pd

from gui.widgets.fairness import render_fairness_board


# ---------------------------------------------------------------------------
# Stage 2 – Data Quality
# ---------------------------------------------------------------------------

def display_quality_results(tool_result: dict):
    """Render data-quality metrics and per-column issue table."""
    if tool_result.get("status") != "success":
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", tool_result.get("total_rows", 0))
    with col2:
        st.metric("Missing Values", tool_result.get("total_missing_values", 0))
    with col3:
        st.metric("Missing %", f"{tool_result.get('overall_missing_percentage', 0):.2f}%")

    if tool_result.get("details"):
        st.markdown("#### Issues by Column")

        issues_data = []
        for detail in tool_result["details"]:
            issues_data.append({
                "Column": detail["column"],
                "Data Type": detail["data_type"],
                "Missing Count": detail["missing_count"],
                "Missing %": f"{detail['missing_percentage']:.2f}%",
                "Issues": detail.get("detected_issues", ""),
            })

        if issues_data:
            df_issues = pd.DataFrame(issues_data)
            st.dataframe(df_issues, width="stretch")


# ---------------------------------------------------------------------------
# Stage 3 – Sensitive Attribute Detection
# ---------------------------------------------------------------------------

def display_sensitive_results(stage_data: dict):
    """Show the list of identified sensitive columns."""
    sensitive_cols = stage_data.get("sensitive_columns", [])

    if sensitive_cols:
        st.markdown(f"**Identified Sensitive Columns:** {', '.join(sensitive_cols)}")
        st.markdown("---")


# ---------------------------------------------------------------------------
# Stage 4 – Imbalance Analysis  (the *redefined* version used in stepwise UI)
# ---------------------------------------------------------------------------

def display_imbalance_results(stage_result: dict):
    """Render imbalance analysis including optional proxy-model board."""
    tool_result = stage_result.get("tool_result", {})
    proxy_results = stage_result.get("proxy_model_results")

    st.info("Class imbalance analysis focuses on detecting skewed distributions in sensitive columns.")

    if tool_result.get("status") == "error":
        st.error(f"Error analyzing imbalance: {tool_result.get('message', 'Unknown error')}")
        return

    if proxy_results and proxy_results.get("status") == "success":
        render_fairness_board(proxy_results, title="Proxy Model Fairness Analysis (Stage 4)")

    if "imbalance_report" in tool_result:
        report = tool_result["imbalance_report"]

        for col, details in report.items():
            if col == "target_column":
                continue
            with st.expander(f"Distribution Details: {col}"):
                st.write(f"Imbalance Ratio: {details.get('imbalance_ratio', 0):.2f}")
                dist = details.get("distribution", {})
                if dist:
                    st.bar_chart(dist)


# ---------------------------------------------------------------------------
# Stage 4.5 – Target Fairness Analysis  (the *redefined* version)
# ---------------------------------------------------------------------------

def display_fairness_results(stage_result: dict):
    """Render target fairness with intersectional proxy + visualizations."""
    tool_result = stage_result.get("tool_result", {})
    intersectional_proxy = stage_result.get("intersectional_proxy_results")

    if tool_result.get("status") == "success":
        st.markdown(f"**Target Column:** {tool_result.get('target_column')}")

        if intersectional_proxy:
            render_fairness_board(
                intersectional_proxy,
                title="Target Fairness Analysis (Stage 4.5 - Intersectional)",
            )

    # --- Visualizations ---
    if stage_result.get("tool_result") and stage_result["tool_result"].get("generated_images"):
        generated_images = stage_result["tool_result"]["generated_images"]
        if generated_images:
            st.markdown("---")
            st.markdown("**Visualizations:**")

            main_images = []
            combination_images: dict[str, list] = {}

            for img_path in generated_images:
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
                        key="main_viz_selector_cleanup",
                    )
                    if selected_main != "None":
                        st.image(main_image_options[selected_main], caption=selected_main, width="stretch")

            if combination_images:
                st.markdown("---")
                st.markdown("#### Combined Sensitive Attribute Analysis")
                combo_options = sorted(combination_images.keys())
                selected_combo = st.selectbox(
                    "Select attribute combination to analyze:",
                    options=["None"] + combo_options,
                    key="combo_selector_cleanup",
                )
                if selected_combo != "None":
                    st.markdown(f"##### {selected_combo}")
                    combo_imgs = combination_images[selected_combo]
                    combo_image_options = {}
                    for img_path in combo_imgs:
                        filename = os.path.basename(img_path)
                        display_name = filename.replace(".png", "").replace("_", " ").title()
                        combo_image_options[display_name] = img_path

                    if combo_image_options:
                        selected_combo_img = st.selectbox(
                            f"Select {selected_combo} visualization:",
                            options=["None"] + list(combo_image_options.keys()),
                            key=f"combo_img_selector_{selected_combo.replace(' + ', '_')}_cleanup",
                        )
                        if selected_combo_img != "None":
                            st.image(
                                combo_image_options[selected_combo_img],
                                caption=selected_combo_img,
                                width="stretch",
                            )
