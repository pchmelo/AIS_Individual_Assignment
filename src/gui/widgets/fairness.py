import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Shared metric glossary (used in help= tooltips)
# ---------------------------------------------------------------------------

_METRIC_HELP = {
    "spd": (
        "Statistical Parity Difference (SPD): the difference between the highest and lowest "
        "positive-prediction rates across demographic groups. "
        "Ideal value: 0 (all groups are equally likely to receive a positive prediction). "
        "Interpretation: closer to 0 → fairer; values above ±0.1 suggest potential bias."
    ),
    "di": (
        "Disparate Impact (DI): the ratio of the lowest to the highest positive-prediction rate "
        "across groups (min_rate / max_rate). "
        "Ideal value: 1.0 (all groups treated equally). "
        "Interpretation: closer to 1 → fairer; below 0.8 triggers the '80% rule' — "
        "the model may be discriminating against a protected group."
    ),
    "accuracy": (
        "Accuracy: proportion of all predictions (positive and negative) that are correct. "
        "Formula: (TP + TN) / (TP + TN + FP + FN). "
        "Ideal value: 1.0. "
        "Interpretation: closer to 1 → more accurate; large differences between groups suggest "
        "the model learns better for some groups than others."
    ),
    "f1": (
        "F1 Score (macro): harmonic mean of precision and recall, averaged across all classes. "
        "Formula: 2 × (Precision × Recall) / (Precision + Recall). "
        "Ideal value: 1.0. "
        "Interpretation: closer to 1 → better balance between false positives and false negatives; "
        "values near 0 indicate poor classification for that group."
    ),
    "selection_rate": (
        "Selection Rate (Positive Rate): proportion of samples in this group predicted as positive "
        "by the model. "
        "Formula: (TP + FP) / total. "
        "Ideal value: equal across all groups (consistent with base rate). "
        "Interpretation: large differences across groups indicate the model "
        "systematically favours or disfavours certain groups."
    ),
    "base_rate": (
        "Base Rate (Actual Positive Rate): proportion of samples in this group that are actually "
        "positive in the ground-truth labels. "
        "Formula: (TP + FN) / total. "
        "Interpretation: reflects real-world prevalence for this group. "
        "Comparing Base Rate to Selection Rate reveals whether the model over- or under-predicts "
        "positive outcomes relative to reality."
    ),
    "fnr": (
        "False Negative Rate (FNR) — also called Miss Rate: proportion of actual positives that "
        "the model incorrectly predicts as negative. "
        "Formula: FN / (TP + FN). "
        "Ideal value: 0 (no actual positives are missed). "
        "Interpretation: closer to 0 → fewer missed positives; high FNR in a group means the model "
        "fails to identify positive outcomes for that group (e.g. loan approvals, medical diagnoses). "
        "Large differences in FNR across groups indicate Equalised Odds violations."
    ),
    "fpr": (
        "False Positive Rate (FPR) — also called Fall-Out: proportion of actual negatives that "
        "the model incorrectly predicts as positive. "
        "Formula: FP / (TN + FP). "
        "Ideal value: 0 (no negatives wrongly labelled positive). "
        "Interpretation: closer to 0 → fewer false alarms; high FPR in a group means the model "
        "over-predicts positive outcomes for that group. "
        "Large differences across groups suggest disparate false-alarm rates."
    ),
    "tpr": (
        "True Positive Rate (TPR) — also called Sensitivity or Recall: proportion of actual positives "
        "correctly identified by the model. "
        "Formula: TP / (TP + FN) = 1 − FNR. "
        "Ideal value: 1.0 (all actual positives are found). "
        "Interpretation: closer to 1 → better recall; low TPR for a group means the model misses "
        "many real positive cases (e.g. it under-approves loans for that group). "
        "Equal TPR across groups satisfies the Equal Opportunity criterion."
    ),
    "tnr": (
        "True Negative Rate (TNR) — also called Specificity: proportion of actual negatives "
        "correctly identified as negative. "
        "Formula: TN / (TN + FP) = 1 − FPR. "
        "Ideal value: 1.0 (all actual negatives are correctly rejected). "
        "Interpretation: closer to 1 → fewer false alarms; low TNR means the model over-predicts "
        "positives. Disparities in TNR between groups can indicate differential treatment."
    ),
    "fnr_ratio": (
        "Max/Min FNR Ratio: ratio of the highest False Negative Rate to the lowest across all groups. "
        "Ideal value: 1.0 (all groups have the same FNR). "
        "Interpretation: closer to 1 → more equitable; above 1.25 is typically considered a "
        "significant disparity — one group is missing far more positive outcomes than another."
    ),
    "imbalance_ratio": (
        "Imbalance Ratio: ratio of the majority class count to the minority class count. "
        "Ideal value: 1.0 (perfectly balanced classes). "
        "Interpretation: closer to 1 → more balanced dataset; values much greater than 1 indicate "
        "that one class dominates the data, which can lead to biased model predictions. "
        "A ratio above 3–4 is generally considered problematic."
    ),
}


def _render_metric_legend():
    """Render a compact metric legend explaining every column in the fairness table."""
    with st.expander("Metric Glossary — click to expand", expanded=False):
        st.markdown(
            """
| Metric | Full Name | Ideal | Direction |
|--------|-----------|-------|-----------|
| **SPD** | Statistical Parity Difference | 0 | Closer to 0 is fairer |
| **DI** | Disparate Impact | 1.0 | Closer to 1 is fairer; < 0.8 triggers 80% rule |
| **Accuracy** | Accuracy | 1.0 | Closer to 1 is better |
| **F1 Score** | F1 Score (macro) | 1.0 | Closer to 1 is better |
| **Selection Rate** | Predicted Positive Rate | Equal across groups | Large differences = potential bias |
| **Base Rate** | Actual Positive Rate | — | Reflects ground-truth prevalence |
| **FNR** | False Negative Rate (Miss Rate) | 0 | Closer to 0 is better; large across-group differences = Equalised Odds violation |
| **FPR** | False Positive Rate (Fall-Out) | 0 | Closer to 0 is better |
| **TPR** | True Positive Rate (Sensitivity / Recall) | 1.0 | Closer to 1 is better |
| **TNR** | True Negative Rate (Specificity) | 1.0 | Closer to 1 is better |

> **How to read the table**: each row is a demographic group. Compare rows to spot
> disparities. Groups with significantly higher FNR, lower TPR, or lower Selection Rate than
> others may be disadvantaged by the model.
            """
        )


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

    # Render glossary once above the tabs
    _render_metric_legend()

    def format_tab_title(attr: str) -> str:
        if attr.endswith("_combined"):
            return attr.replace("_combined", "").replace("_", " + ").title()
        return attr.replace("_", " ").title()

    tabs = st.tabs([format_tab_title(col) for col in fairness.keys()])

    for idx, (col_name, data) in enumerate(fairness.items()):
        with tabs[idx]:
            metrics_data = data.get("metrics", {})

            # ── Top-level fairness metrics ──
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                spd_value = metrics_data.get("statistical_parity_difference", 0)
                st.metric(
                    "Stat Parity Diff (SPD)",
                    f"{spd_value:.4f}",
                    help=_METRIC_HELP["spd"],
                )
            with m_col2:
                di_value = metrics_data.get("disparate_impact", 0)
                st.metric(
                    "Disparate Impact (DI)",
                    f"{di_value:.4f}",
                    help=_METRIC_HELP["di"],
                )
            with m_col3:
                max_group = metrics_data.get("max_positive_rate_group", "N/A")
                min_group = metrics_data.get("min_positive_rate_group", "N/A")
                if max_group != "N/A" and min_group != "N/A":
                    st.caption(f"**Highest Rate:** {max_group}")
                    st.caption(f"**Lowest Rate:** {min_group}")

            # ── Per-group metrics table ──
            groups_data = data.get("groups", {})
            rows = []
            for group, metrics in groups_data.items():
                rows.append(
                    {
                        "Group": group.replace("_", " + "),
                        "Count": metrics.get("count"),
                        "Accuracy": metrics.get("accuracy"),
                        "F1 Score": metrics.get("f1_macro"),
                        "Selection Rate": metrics.get("positive_rate"),
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
                    "Count": st.column_config.NumberColumn(
                        "Count",
                        help="Number of samples in this demographic group.",
                        format="%d",
                    ),
                    "Accuracy": st.column_config.ProgressColumn(
                        "Accuracy",
                        help=_METRIC_HELP["accuracy"],
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "F1 Score": st.column_config.ProgressColumn(
                        "F1 Score",
                        help=_METRIC_HELP["f1"],
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Selection Rate": st.column_config.ProgressColumn(
                        "Selection Rate",
                        help=_METRIC_HELP["selection_rate"],
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Base Rate": st.column_config.ProgressColumn(
                        "Base Rate",
                        help=_METRIC_HELP["base_rate"],
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "FNR": st.column_config.NumberColumn(
                        "FNR",
                        format="%.4f",
                        help=_METRIC_HELP["fnr"],
                    ),
                    "FPR": st.column_config.NumberColumn(
                        "FPR",
                        format="%.4f",
                        help=_METRIC_HELP["fpr"],
                    ),
                    "TPR": st.column_config.NumberColumn(
                        "TPR",
                        format="%.4f",
                        help=_METRIC_HELP["tpr"],
                    ),
                    "TNR": st.column_config.NumberColumn(
                        "TNR",
                        format="%.4f",
                        help=_METRIC_HELP["tnr"],
                    ),
                },
            )

            # ── FNR ratio indicator ──
            fnrs = [r["FNR"] for r in rows if isinstance(r["FNR"], (int, float))]
            if fnrs and len(fnrs) > 1 and min(fnrs) > 0:
                ratio = max(fnrs) / min(fnrs)
                st.metric(
                    "Max/Min FNR Ratio",
                    f"{ratio:.2f}",
                    help=_METRIC_HELP["fnr_ratio"],
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

    # Brief guidance
    st.caption(
        "SPD (Statistical Parity Difference): ideal = 0, closer to 0 is fairer. "
        "DI (Disparate Impact): ideal = 1.0, closer to 1 is fairer; below 0.8 may indicate discrimination."
    )

    st.markdown("---")

    def format_tab_title(attr: str) -> str:
        if attr.endswith("_combined"):
            return attr.replace("_combined", "").replace("_", " + ").title()
        return attr.replace("_", " ").title()

    attr_tabs = st.tabs([format_tab_title(attr) for attr in per_attr_comparison.keys()])

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
                st.metric(
                    "Stat Parity Diff",
                    f"{spd_baseline:.4f}",
                    help=(
                        "SPD before mitigation. "
                        "Ideal: 0. Closer to 0 is fairer. "
                        "Values above ±0.1 suggest potential bias."
                    ),
                )
                st.metric(
                    "Disparate Impact",
                    f"{di_baseline:.4f}",
                    help=(
                        "DI before mitigation. "
                        "Ideal: 1.0. Closer to 1 is fairer. "
                        "Below 0.8 triggers the 80% rule (potential discrimination)."
                    ),
                )

            with col2:
                st.markdown(f"**After {method_name}**")
                st.metric(
                    "Stat Parity Diff",
                    f"{spd_mitigated:.4f}",
                    help=(
                        "SPD after mitigation. "
                        "Ideal: 0. A decrease means the groups are more equally treated."
                    ),
                )
                st.metric(
                    "Disparate Impact",
                    f"{di_mitigated:.4f}",
                    help=(
                        "DI after mitigation. "
                        "Ideal: 1.0. An increase toward 1.0 means improved equity."
                    ),
                )

            with col3:
                st.markdown("**Change**")
                if spd_improved:
                    st.metric(
                        "SPD Change", f"{spd_change:+.4f}",
                        delta=f"↓ {abs(spd_change):.4f}", delta_color="normal",
                        help=(
                            "SPD decreased (improved): the group selection rates are now closer "
                            "together. A negative SPD change is good — it means less disparity."
                        ),
                    )
                else:
                    st.metric(
                        "SPD Change", f"{spd_change:+.4f}",
                        delta=f"↑ {abs(spd_change):.4f}", delta_color="inverse",
                        help=(
                            "SPD increased (degraded): the gap between group selection rates grew. "
                            "This mitigation method may have worsened statistical parity for this attribute."
                        ),
                    )
                if di_improved:
                    st.metric(
                        "DI Change", f"{di_change:+.4f}",
                        delta=f"↑ {abs(di_change):.4f}", delta_color="normal",
                        help=(
                            "DI increased toward 1.0 (improved): the ratio between group positive "
                            "rates is now closer to equal. A positive DI change is good."
                        ),
                    )
                else:
                    st.metric(
                        "DI Change", f"{di_change:+.4f}",
                        delta=f"↓ {abs(di_change):.4f}", delta_color="inverse",
                        help=(
                            "DI decreased away from 1.0 (degraded): the positive-rate ratio between "
                            "groups worsened. This mitigation method may have increased disparity for "
                            "this attribute."
                        ),
                    )

            st.markdown("---")
            st.markdown("**Interpretation:**")

            improvements = []
            degradations = []
            if spd_improved:
                improvements.append(
                    f"Statistical Parity improved by {abs(spd_change):.4f} "
                    f"(SPD moved closer to 0 — groups are treated more equally)."
                )
            else:
                degradations.append(
                    f"Statistical Parity degraded by {abs(spd_change):.4f} "
                    f"(SPD moved further from 0 — disparity increased)."
                )
            if di_improved:
                improvements.append(
                    f"Disparate Impact improved by {abs(di_change):.4f} "
                    f"(DI moved closer to 1.0 — positive-rate ratio is more equitable)."
                )
            else:
                degradations.append(
                    f"Disparate Impact degraded by {abs(di_change):.4f} "
                    f"(DI moved further from 1.0 — positive-rate ratio became less equitable)."
                )

            for line in improvements:
                st.success(line)
            for line in degradations:
                st.warning(line)

            # ── Per-group comparison — two selectbox-driven views ────────
            group_comparison = attr_data.get("group_comparison", [])
            if group_comparison:
                import pandas as pd

                # Shared metric registry (key, display label, higher=better, full help text)
                _ALL_METRICS = [
                    ("accuracy",      "Accuracy",       True,  _METRIC_HELP["accuracy"]),
                    ("f1_macro",      "F1 Score",       True,  _METRIC_HELP["f1"]),
                    ("positive_rate", "Selection Rate", True,  _METRIC_HELP["selection_rate"]),
                    ("base_rate",     "Base Rate",       None, _METRIC_HELP["base_rate"]),
                    ("fnr",           "FNR",             False, _METRIC_HELP["fnr"]),
                    ("fpr",           "FPR",             False, _METRIC_HELP["fpr"]),
                    ("tpr",           "TPR",             True,  _METRIC_HELP["tpr"]),
                    ("tnr",           "TNR",             True,  _METRIC_HELP["tnr"]),
                ]

                with st.expander("Detailed Group Metrics & Comparisons", expanded=False):
                    # ════════════════════════════════════════════════════════
                    # View 1 – choose a METRIC → compare all groups
                    # ════════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown("#### Compare Groups by Metric")
                    st.caption(
                        "Pick a fairness metric below. The table shows every demographic group's "
                        "value **before** and **after** mitigation, plus the change. "
                        "Hover over any column header for a full definition."
                    )

                    sel_metric_label = st.selectbox(
                        "Select metric to view across groups:",
                        options=[m[1] for m in _ALL_METRICS],
                        key=f"metric_sel_{idx}_{method_name}",
                    )
                    sel_key, _, higher_better, sel_help = next(
                        m for m in _ALL_METRICS if m[1] == sel_metric_label
                    )
                    ideal_val = (
                        "1.0 (higher is better)" if higher_better is True
                        else "0.0 (lower is better)" if higher_better is False
                        else "Equal across groups"
                    )
                    st.info(
                        f"**{sel_metric_label}** — {sel_help[:200]}{'…' if len(sel_help) > 200 else ''}  "
                        f"  \n**Ideal:** {ideal_val}"
                    )

                    rows_v1 = []
                    for grp in group_comparison:
                        bv = grp.get(f"baseline_{sel_key}")
                        mv = grp.get(f"mitigated_{sel_key}")
                        dv = grp.get(f"delta_{sel_key}")
                        row = {"Group": grp["group"].replace("_", " + ")}
                        row["Before (Stage 4)"] = round(float(bv), 4) if bv is not None else None
                        row["After (Mitigated)"] = round(float(mv), 4) if mv is not None else None
                        if dv is not None and higher_better is not None:
                            arrow = "↑" if dv > 0 else ("↓" if dv < 0 else "=")
                            better = (higher_better and dv > 0) or (not higher_better and dv < 0)
                            row["Δ Change"] = f"{'✓' if better else '✗'} {arrow}{abs(dv):.4f}"
                        elif dv is not None:
                            row["Δ Change"] = f"{dv:+.4f}"
                        else:
                            row["Δ Change"] = "—"
                        rows_v1.append(row)

                    if rows_v1:
                        _delta_help_v1 = (
                            f"Change in {sel_metric_label} (After − Before). "
                            + ("✓ = improved (↑ towards 1); ✗ = degraded." if higher_better is True
                               else ("✓ = improved (↓ towards 0); ✗ = degraded." if higher_better is False
                                     else "Neutral metric — no single ideal direction."))
                        )
                        st.dataframe(
                            pd.DataFrame(rows_v1),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Group": st.column_config.TextColumn(
                                    "Group",
                                    help="Demographic group. Each row represents a distinct subpopulation within this sensitive attribute.",
                                ),
                                "Before (Stage 4)": st.column_config.NumberColumn(
                                    "Before (Stage 4)",
                                    format="%.4f",
                                    help=f"{sel_metric_label} for this group in the original dataset (Stage 4 baseline), before any mitigation was applied.",
                                ),
                                "After (Mitigated)": st.column_config.NumberColumn(
                                    "After (Mitigated)",
                                    format="%.4f",
                                    help=f"{sel_metric_label} for this group after applying {method_name}. Compare to 'Before' to see the effect of mitigation.",
                                ),
                                "Δ Change": st.column_config.TextColumn(
                                    "Δ Change",
                                    help=_delta_help_v1,
                                ),
                            },
                        )

                    # ════════════════════════════════════════════════════════
                    # View 2 – choose a GROUP → see all metrics
                    # ════════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown("#### All Metrics for a Selected Group")
                    st.caption(
                        "Pick a demographic group below. The table shows **all** fairness and "
                        "performance metrics for that group — before and after mitigation — "
                        "along with definitions and ideal values."
                    )

                    group_names = [g["group"].replace("_", " + ") for g in group_comparison]
                    selected_group = st.selectbox(
                        "Select demographic group:",
                        options=group_names,
                        key=f"group_sel_{idx}_{method_name}",
                    )
                    sel_grp_raw = next(
                        (g for g in group_comparison if g["group"].replace("_", " + ") == selected_group),
                        None,
                    )

                    if sel_grp_raw:
                        rows_v2 = []
                        for m_key, m_label, m_better, m_help in _ALL_METRICS:
                            bv = sel_grp_raw.get(f"baseline_{m_key}")
                            mv = sel_grp_raw.get(f"mitigated_{m_key}")
                            dv = sel_grp_raw.get(f"delta_{m_key}")
                            ideal = (
                                "1.0 (higher is better)" if m_better is True
                                else "0.0 (lower is better)" if m_better is False
                                else "Equal across groups"
                            )
                            bv_str = f"{float(bv):.4f}" if bv is not None else "—"
                            mv_str = f"{float(mv):.4f}" if mv is not None else "—"
                            if dv is not None and m_better is not None:
                                arrow = "↑" if dv > 0 else ("↓" if dv < 0 else "=")
                                better = (m_better and dv > 0) or (not m_better and dv < 0)
                                change_str = f"{'✓ Improved' if better else '✗ Degraded'} ({arrow}{abs(dv):.4f})"
                            elif dv is not None:
                                change_str = f"{dv:+.4f}"
                            else:
                                change_str = "—"
                            rows_v2.append({
                                "Metric": m_label,
                                "Ideal Value": ideal,
                                "Before (Stage 4)": bv_str,
                                "After (Mitigated)": mv_str,
                                "Change": change_str,
                                "_help": m_help,
                            })

                        df_v2 = pd.DataFrame(rows_v2)
                        # Drop internal help column before display
                        st.dataframe(
                            df_v2.drop(columns=["_help"]),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Metric": st.column_config.TextColumn(
                                    "Metric",
                                    width="small",
                                    help="Fairness or model performance metric. Hover each row's tooltip for a full definition.",
                                ),
                                "Ideal Value": st.column_config.TextColumn(
                                    "Ideal Value",
                                    width="medium",
                                    help="The target value this metric should reach in a perfectly fair, accurate model.",
                                ),
                                "Before (Stage 4)": st.column_config.TextColumn(
                                    "Before (Stage 4)",
                                    help=f"Value for the '{selected_group}' group in the original dataset, before {method_name} was applied.",
                                ),
                                "After (Mitigated)": st.column_config.TextColumn(
                                    "After (Mitigated)",
                                    help=f"Value for the '{selected_group}' group after applying {method_name}.",
                                ),
                                "Change": st.column_config.TextColumn(
                                    "Change",
                                    help=(
                                        "✓ Improved = the metric moved toward its ideal value after mitigation. "
                                        "✗ Degraded = the metric moved away from its ideal value. "
                                        "The arrow (↑/↓) shows the direction of change."
                                    ),
                                ),
                            },
                        )
                        # Per-metric definitions expandable
                        with st.expander("Metric definitions", expanded=False):
                            for m_key, m_label, m_better, m_help in _ALL_METRICS:
                                st.markdown(f"**{m_label}**: {m_help}")
                                st.markdown("---")

# ---------------------------------------------------------------------------
# Mitigation Tables (Report-like views)
# ---------------------------------------------------------------------------

def render_mitigated_fairness_table(mitigated_metrics: dict, title: str = "Evaluated Fairness Metrics (After Mitigation)"):
    """Render the summarized table of fairness metrics after mitigation."""
    if not mitigated_metrics or mitigated_metrics.get("status") != "success":
        return

    fairness = mitigated_metrics.get("fairness_analysis", {})
    if not fairness:
        return
        
    st.markdown(f"#### {title}")
    
    rows = []
    for col_name, data in fairness.items():
        metrics_data = data.get("metrics", {})
        spd_value = metrics_data.get("statistical_parity_difference", 0)
        di_value = metrics_data.get("disparate_impact", 0)
        max_group = metrics_data.get("max_positive_rate_group", "N/A")
        min_group = metrics_data.get("min_positive_rate_group", "N/A")

        attr_display = col_name.replace("_combined", "").replace("_", " + ")
        
        rows.append({
            "Sensitive Attribute": attr_display,
            "Stat Parity Diff": spd_value,
            "Disparate Impact": di_value,
            "Highest Rate Group": str(max_group),
            "Lowest Rate Group": str(min_group)
        })
        
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Stat Parity Diff": st.column_config.NumberColumn("Stat Parity Diff", format="%.4f"),
                "Disparate Impact": st.column_config.NumberColumn("Disparate Impact", format="%.4f")
            }
        )

def render_mitigation_scorecard(fairness_comparison: dict, comparison_result: dict):
    """Render the unified Mitigation Scorecard table."""
    rows = []
    
    if comparison_result:
        imb = comparison_result.get("imbalance_metrics", {})
        if imb:
            orig_ratio = float(imb.get("original_imbalance_ratio", 0))
            new_ratio = float(imb.get("mitigated_imbalance_ratio", 0))
            diff = new_ratio - orig_ratio
            improvement = imb.get("improvement", "No")
            rows.append({
                "Metric": "Imbalance Ratio",
                "Before Mitigation": orig_ratio,
                "After Mitigation": new_ratio,
                "Improved?": improvement,
                "Diff": diff
            })
            
    if fairness_comparison:
        per_attr = fairness_comparison.get("per_attribute_comparison", {})
        for attr, metrics in per_attr.items():
            attr_display = attr.replace("_combined", "").replace("_", " + ")
            spd = metrics.get("statistical_parity_difference", {})
            if spd:
                sb = float(spd.get("baseline", 0))
                sm = float(spd.get("mitigated", 0))
                si = "Yes" if spd.get("improved") else "No"
                rows.append({
                    "Metric": f"{attr_display} (Stat Parity)",
                    "Before Mitigation": sb,
                    "After Mitigation": sm,
                    "Improved?": si,
                    "Diff": float(spd.get("change", 0))
                })
                
            di = metrics.get("disparate_impact", {})
            if di:
                db = float(di.get("baseline", 0))
                dm = float(di.get("mitigated", 0))
                di_imp = "Yes" if di.get("improved") else "No"
                rows.append({
                    "Metric": f"{attr_display} (Disp Impact)",
                    "Before Mitigation": db,
                    "After Mitigation": dm,
                    "Improved?": di_imp,
                    "Diff": float(di.get("change", 0))
                })
                
    if rows:
        st.markdown("#### Mitigation Scorecard")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Before Mitigation": st.column_config.NumberColumn("Before Mitigation", format="%.4f"),
                "After Mitigation": st.column_config.NumberColumn("After Mitigation", format="%.4f"),
                "Diff": st.column_config.NumberColumn("Diff", format="%+.4f"),
            }
        )


