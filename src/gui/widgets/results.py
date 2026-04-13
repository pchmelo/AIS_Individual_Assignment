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
    """Render imbalance analysis including optional ML model board."""
    tool_result = stage_result.get("tool_result", {})
    ml_results = stage_result.get("ml_model_results")

    st.info(
        "Class imbalance analysis detects skewed value distributions in sensitive attributes. "
        "A balanced attribute has roughly equal counts in each category; an imbalanced one "
        "has one group dominating the data, which can distort model fairness."
    )

    if tool_result.get("status") == "error":
        st.error(f"Error analyzing imbalance: {tool_result.get('message', 'Unknown error')}")
        return

    if ml_results and ml_results.get("status") == "success":
        render_fairness_board(ml_results, title="ML Model Fairness Analysis (Stage 4)")

    if "imbalance_report" in tool_result:
        report = tool_result["imbalance_report"]

        st.markdown("#### Sensitive Attribute Distributions")
        st.caption(
            "**Imbalance Ratio** = majority class count ÷ minority class count. "
            "Ideal value: 1.0 (perfectly balanced). "
            "Interpretation: closer to 1 → more balanced; ratios above 3–4 are considered high "
            "and can cause the model to under-represent minority groups in its predictions."
        )

        for col, details in report.items():
            if col == "target_column":
                continue
            ratio = details.get("imbalance_ratio", 0)
            if ratio >= 4:
                ratio_label = f"{ratio:.2f} — High imbalance"
            elif ratio >= 2:
                ratio_label = f"{ratio:.2f} — Moderate imbalance"
            else:
                ratio_label = f"{ratio:.2f} — Low imbalance (near-balanced)"
            with st.expander(f"Distribution Details: {col}"):
                st.metric(
                    "Imbalance Ratio",
                    ratio_label,
                    help=(
                        "Imbalance Ratio = count of largest class ÷ count of smallest class. "
                        "Ideal: 1.0. Closer to 1 means the groups are similar in size. "
                        "A high ratio (> 3) means one group vastly outnumbers others, "
                        "which can lead to biased predictions favouring the majority group."
                    ),
                )
                dist = details.get("distribution", {})
                if dist:
                    st.bar_chart(dist)


# ---------------------------------------------------------------------------
# Stage 4.5 – Target Fairness Analysis 
# ---------------------------------------------------------------------------

def display_fairness_results(stage_result: dict):
    """Render target fairness with intersectional ML + visualizations."""
    tool_result = stage_result.get("tool_result", {})
    intersectional_ml = stage_result.get("intersectional_ml_results")

    if tool_result.get("status") != "success":
        # Show error/skip message if something went wrong
        msg = tool_result.get("message", "No fairness data available.")
        st.warning(f"Stage 4.5: {msg}")
        return

    target_col = tool_result.get("target_column", "—")
    st.markdown(f"**Target Column:** `{target_col}`")

    # ── Intersectional ML board (shown if ML model was run) ──────────
    if intersectional_ml and intersectional_ml.get("status") == "success":
        render_fairness_board(
            intersectional_ml,
            title="Target Fairness Analysis (Stage 4.5 - Intersectional ML)",
        )

    # ── Per-attribute group breakdown (always shown) ─────────────────
    target_rates_by_group = tool_result.get("target_rates_by_group", {})
    if target_rates_by_group:
        st.markdown("---")
        st.markdown("#### Target Rate by Sensitive Attribute Group")
        st.caption(
            "For each sensitive attribute, the table below shows how the target variable "
            "is distributed across demographic groups. "
            "Look for large differences in the positive-class percentage between groups — "
            "these indicate potential disparate impact (unfair outcomes)."
        )

        for attr, groups in target_rates_by_group.items():
            with st.expander(f"**{attr}** — Target Rate by Group", expanded=True):
                rows = []
                for group_val, group_data in groups.items():
                    total = group_data.get("total_count", 0)
                    pcts = group_data.get("target_percentages", {})
                    dist = group_data.get("target_distribution", {})
                    # Build a row with count + pct per target class
                    row = {"Group": str(group_val), "Count": total}
                    for cls, pct in pcts.items():
                        row[f"{cls} (%)"] = round(float(pct), 2)
                    for cls, cnt in dist.items():
                        row[f"{cls} (count)"] = int(cnt)
                    rows.append(row)

                if rows:
                    df_group = pd.DataFrame(rows).sort_values("Count", ascending=False)
                    st.dataframe(df_group, hide_index=True, width="stretch")

                    # Highlight disparity
                    pct_cols = [c for c in df_group.columns if c.endswith("(%)")]
                    if pct_cols:
                        # Use first percentage column (positive class) for disparity check
                        primary_pct_col = pct_cols[0]
                        try:
                            vals = df_group[primary_pct_col].dropna().tolist()
                            if len(vals) >= 2:
                                gap = max(vals) - min(vals)
                                if gap > 10:
                                    st.warning(
                                        f"Large disparity detected: the '{primary_pct_col}' rate "
                                        f"varies by **{gap:.1f} percentage points** across groups — "
                                        f"potential fairness concern."
                                    )
                                else:
                                    st.success(
                                        f"Low disparity: '{primary_pct_col}' varies by only "
                                        f"{gap:.1f} pp across groups."
                                    )
                        except Exception:
                            pass

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
                    if combo_folder and os.path.exists(img_path):
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
                        img_path = main_image_options[selected_main]
                        if os.path.exists(img_path):
                            st.image(img_path, caption=selected_main, width="stretch")
                        else:
                            st.warning(f"Image file not found: {os.path.basename(img_path)}")

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
                        if os.path.exists(img_path):
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
                            img_path = combo_image_options[selected_combo_img]
                            if os.path.exists(img_path):
                                st.image(img_path, caption=selected_combo_img, width="stretch")
                            else:
                                st.warning(f"Image file not found: {os.path.basename(img_path)}")
                    else:
                        st.info(
                            "No visualizations available for this combination. "
                            "Some images may not have been saved (e.g. due to long file path names)."
                        )
