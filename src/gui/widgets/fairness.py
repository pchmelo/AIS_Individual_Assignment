import streamlit as st
import pandas as pd


def render_fairness_board(ml_results: dict, title: str = "Fairness Metrics Board"):
    """Render an interactive fairness metrics board from ML model results."""
    if not ml_results or ml_results.get("status") != "success":
        return

    st.markdown(f"### {title}")
    st.info(
        f"Model: {ml_results.get('model_type')} | "
        f"Test Size: {ml_results.get('test_size')} | "
        f"Accuracy: {ml_results.get('performance', {}).get('accuracy')}"
    )

    fairness = ml_results.get("fairness_analysis", {})
    if not fairness:
        return

    tabs = st.tabs(
        [col.replace("_combined", "").replace("_", " + ") for col in fairness.keys()]
    )

    for idx, (col_name, data) in enumerate(fairness.items()):
        with tabs[idx]:
            metrics_data = data.get("metrics", {})
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                spd_value = metrics_data.get("statistical_parity_difference", 0)
                st.metric(
                    "Stat Parity Diff",
                    f"{spd_value:.4f}",
                    help="Difference between max and min selection rates across groups. Closer to 0 is better. >0.1 may indicate bias.",
                )
            with m_col2:
                di_value = metrics_data.get("disparate_impact", 0)
                st.metric(
                    "Disparate Impact",
                    f"{di_value:.4f}",
                    help="Ratio of min to max selection rate. Should be close to 1.0. <0.8 indicates potential discrimination (80% rule).",
                )
            with m_col3:
                max_group = metrics_data.get("max_positive_rate_group", "N/A")
                min_group = metrics_data.get("min_positive_rate_group", "N/A")
                if max_group != "N/A" and min_group != "N/A":
                    st.caption(f"**Highest Rate:** {max_group}")
                    st.caption(f"**Lowest Rate:** {min_group}")

            groups_data = data.get("groups", {})
            rows = []
            for group, metrics in groups_data.items():
                rows.append(
                    {
                        "Group": group.replace("_", " + "),
                        "Count": metrics.get("count"),
                        "Accuracy": metrics.get("accuracy"),
                        "F1 Score": metrics.get("f1_macro"),
                        "Pos Rate": metrics.get("positive_rate"),
                        "Base Rate": metrics.get("base_rate"),
                        "FNR": metrics.get("fnr", "N/A"),
                        "FPR": metrics.get("fpr", "N/A"),
                        "TPR": metrics.get("tpr", "N/A"),
                        "TNR": metrics.get("tnr", "N/A"),
                    }
                )

            display_df = pd.DataFrame(rows)
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Group": st.column_config.TextColumn("Group", width="medium"),
                    "Count": st.column_config.NumberColumn("Count", help="Number of samples in this group", format="%d"),
                    "Accuracy": st.column_config.ProgressColumn("Accuracy", help="Proportion of correct predictions", format="%.2f", min_value=0, max_value=1),
                    "F1 Score": st.column_config.ProgressColumn("F1 Score", help="Harmonic mean of precision and recall", format="%.2f", min_value=0, max_value=1),
                    "Pos Rate": st.column_config.ProgressColumn("Selection Rate", help="Predicted Positive Rate", format="%.2f", min_value=0, max_value=1),
                    "Base Rate": st.column_config.ProgressColumn("Base Rate", help="Actual Positive Rate in Test Data", format="%.2f", min_value=0, max_value=1),
                    "FNR": st.column_config.NumberColumn("FNR", format="%.4f", help="False Negative Rate"),
                    "FPR": st.column_config.NumberColumn("FPR", format="%.4f", help="False Positive Rate"),
                    "TPR": st.column_config.NumberColumn("TPR", format="%.4f", help="True Positive Rate (Sensitivity)"),
                    "TNR": st.column_config.NumberColumn("TNR", format="%.4f", help="True Negative Rate (Specificity)"),
                },
            )

            fnrs = [r["FNR"] for r in rows if isinstance(r["FNR"], (int, float))]
            if fnrs and len(fnrs) > 1 and min(fnrs) > 0:
                ratio = max(fnrs) / min(fnrs)
                st.metric(
                    "Max/Min FNR Ratio",
                    f"{ratio:.2f}",
                    help="Ratio of highest to lowest False Negative Rate. > 1.25 indicates significant disparity.",
                )

            f1s = [r["F1 Score"] for r in rows if isinstance(r["F1 Score"], (int, float))]
            if f1s:
                st.caption(f"F1 Range: {min(f1s):.2f} - {max(f1s):.2f}")


# ---------------------------------------------------------------------------
# Before / After fairness comparison board
# ---------------------------------------------------------------------------

def render_fairness_comparison_board(comparison_data: dict, method_name: str = "Mitigation Method"):
    """Render a before/after fairness comparison dashboard."""
    if not comparison_data or comparison_data.get("status") == "error":
        st.warning(
            f"Fairness comparison not available: {comparison_data.get('message', 'Unknown error')}"
        )
        return

    per_attr_comparison = comparison_data.get("per_attribute_comparison", {})
    if not per_attr_comparison:
        st.info(f"No fairness comparison data available for {method_name}")
        return

    st.markdown(f"### Fairness Metrics Comparison: {method_name}")

    overall_improvement = comparison_data.get("overall_improvement", "Unknown")
    _badge = {
        "Significant": st.success,
        "Moderate": st.info,
        "Minor": st.warning,
    }
    badge_fn = _badge.get(overall_improvement, st.error)
    badge_fn(f"**Overall Assessment**: {overall_improvement} Improvement")

    st.markdown("---")

    attr_tabs = st.tabs(
        [attr.replace("_", " ").title() for attr in per_attr_comparison.keys()]
    )

    for idx, (attr_name, attr_data) in enumerate(per_attr_comparison.items()):
        with attr_tabs[idx]:
            st.markdown(f"#### {attr_name.replace('_', ' ').title()}")

            spd_data = attr_data.get("statistical_parity_difference", {})
            spd_baseline = spd_data.get("baseline", 0)
            spd_mitigated = spd_data.get("mitigated", 0)
            spd_change = spd_data.get("change", 0)
            spd_improved = spd_data.get("improved", False)

            di_data = attr_data.get("disparate_impact", {})
            di_baseline = di_data.get("baseline", 0)
            di_mitigated = di_data.get("mitigated", 0)
            di_change = di_data.get("change", 0)
            di_improved = di_data.get("improved", False)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Baseline (Original)**")
                st.metric("Stat Parity Diff", f"{spd_baseline:.4f}", help="Target: 0")
                st.metric("Disparate Impact", f"{di_baseline:.4f}", help="Target: 1.0 (≥0.8 acceptable)")

            with col2:
                st.markdown(f"**After {method_name}**")
                st.metric("Stat Parity Diff", f"{spd_mitigated:.4f}", help="Target: 0")
                st.metric("Disparate Impact", f"{di_mitigated:.4f}", help="Target: 1.0 (≥0.8 acceptable)")

            with col3:
                st.markdown("**Change**")
                if spd_improved:
                    st.metric(
                        "SPD Change", f"{spd_change:+.4f}",
                        delta=f"↓ {abs(spd_change):.4f}", delta_color="normal",
                        help="Positive change = improvement (closer to 0)",
                    )
                else:
                    st.metric(
                        "SPD Change", f"{spd_change:+.4f}",
                        delta=f"↑ {abs(spd_change):.4f}", delta_color="inverse",
                        help="Negative change = degradation (farther from 0)",
                    )
                if di_improved:
                    st.metric(
                        "DI Change", f"{di_change:+.4f}",
                        delta=f"↑ {abs(di_change):.4f}", delta_color="normal",
                        help="Positive change = improvement (closer to 1.0)",
                    )
                else:
                    st.metric(
                        "DI Change", f"{di_change:+.4f}",
                        delta=f"↓ {abs(di_change):.4f}", delta_color="inverse",
                        help="Negative change = degradation (farther from 1.0)",
                    )

            st.markdown("---")
            st.markdown("**Interpretation:**")

            improvements = []
            degradations = []
            if spd_improved:
                improvements.append(f"Statistical Parity improved by {abs(spd_change):.4f}")
            else:
                degradations.append(f"Statistical Parity degraded by {abs(spd_change):.4f}")
            if di_improved:
                improvements.append(f"Disparate Impact improved by {abs(di_change):.4f}")
            else:
                degradations.append(f"Disparate Impact degraded by {abs(di_change):.4f}")

            for line in improvements:
                st.markdown(line)
            for line in degradations:
                st.markdown(line)
