"""
Pipeline utility functions and shared constants.

Centralises markdown-formatting helpers and lookup tables that were
previously scattered across pipeline.py and evaluator.py.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ── Mitigation-technique display-name mapping ──────────────────────────
# Canonical mapping from user-facing / config keys to display names.
# Used by both the CLI evaluator and the GUI to build mitigation configs.
TECHNIQUE_DISPLAY: Dict[str, str] = {
    "reweighting": "Reweighting",
    "resampling": "SMOTE",
    "smote": "SMOTE",
    "oversampling": "Random Oversampling",
    "undersampling": "Random Undersampling",
}


# ── Markdown report helpers ────────────────────────────────────────────

def format_mitigation_markdown(lines: List[str], stage_data: Dict[str, Any]) -> None:
    """Format bias mitigation section as markdown (appends to *lines*)."""
    methods_results = stage_data.get("methods", {})
    applied = stage_data.get("applied_methods", list(methods_results.keys()))

    lines.append(f"**Status:** {stage_data.get('status', 'unknown')}")
    lines.append(f"**Applied Methods:** {', '.join(applied)}")
    lines.append("")

    for method in applied:
        mr = methods_results.get(method, {})
        lines.append(f"### {method.replace('_', ' ').title()}")
        lines.append("")

        if mr.get("status") == "error":
            lines.append(f"**Error:** {mr.get('error', 'Unknown error')}")
            lines.append("")
            continue

        mitigation_result = mr.get("mitigation_result", {})
        if mitigation_result:
            lines.append("#### Mitigation Results")
            lines.append("")
            if "method" in mitigation_result:
                lines.append(f"- **Technique:** {mitigation_result['method']}")
            if "original_rows" in mitigation_result and "new_rows" in mitigation_result:
                orig = mitigation_result["original_rows"]
                new = mitigation_result["new_rows"]
                change = new - orig
                pct = (change / orig * 100) if orig > 0 else 0
                lines.append(f"- **Dataset Size:** {orig:,} → {new:,} ({pct:+.1f}%)")
            if "rows_added" in mitigation_result:
                lines.append(f"- **Samples Added:** +{mitigation_result['rows_added']:,}")
            lines.append("")

        fairness_comparison = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
        
        if fairness_comparison:
            mitigated_metrics = fairness_comparison.get("mitigated_metrics", {})
            if mitigated_metrics and mitigated_metrics.get("status") == "success":
                format_ml_model_markdown(lines, mitigated_metrics, title=f"Evaluation ML Model ({method})")

        comparison_result = mr.get("comparison_result") or mitigation_result.get("comparison_result")
        if comparison_result:
            imb = comparison_result.get("imbalance_metrics", {})
            if imb:
                lines.append("#### Imbalance Improvement")
                lines.append("")
                lines.append(f"- **Original Ratio:** {imb.get('original_imbalance_ratio', 'N/A'):.2f}")
                lines.append(f"- **Mitigated Ratio:** {imb.get('mitigated_imbalance_ratio', 'N/A'):.2f}")
                improvement = imb.get("improvement", "No")
                lines.append(f"- **Improved:** {improvement}")
                lines.append("")

            if "agent_analysis" in comparison_result:
                lines.append("#### Agent Analysis")
                lines.append("")
                lines.append(comparison_result["agent_analysis"])
                lines.append("")


def format_pair_selection_markdown(lines: List[str], pair_selection: Dict[str, Any]) -> None:
    """Format pair selection info as markdown (appends to *lines*)."""
    lines.append("### Intersectional Pair Selection")
    lines.append("")
    lines.append(f"**Max Pairs Limit:** {pair_selection.get('max_pairs_limit', 'N/A')}")
    lines.append(f"**Total Possible Pairs:** {pair_selection.get('total_possible_pairs', 'N/A')}")
    lines.append("")
    lines.append("**Selected Pairs for Analysis:**")
    for pair in pair_selection.get("selected_pairs", []):
        lines.append(f"- {pair}")
    lines.append("")
    lines.append("**Selection Reasoning:**")
    lines.append("")
    lines.append(pair_selection.get("reasoning", "No reasoning provided."))
    lines.append("")


def _get_csv_folder_and_prefix(title: str):
    """Determine the subfolder name and file prefix for exported CSVs."""
    title_lower = title.lower()
    if "base" in title_lower:
        return "base_fairness", "base"
    elif "intersectional" in title_lower:
        return "intersectional_fairness", "intersectional"
    elif "evaluation" in title_lower:
        import re
        m = re.search(r'\((.*?)\)', title)
        method = m.group(1).lower().replace(" ", "_") if m else "unknown"
        return f"mitigation_{method}", f"mitigated_{method}"
    else:
        prefix = title_lower.replace(" ", "_").replace("(", "").replace(")", "")
        return "other_fairness", prefix


def format_fairness_board_markdown(lines: List[str], ml_results: Dict[str, Any], title: str = "") -> None:
    """Format the fairness evaluation metrics into a markdown table, omitting raw data."""
    fairness = ml_results.get("fairness_analysis", {})
    if not fairness:
        return

    folder_name, prefix = _get_csv_folder_and_prefix(title)

    for col_name, data in fairness.items():
        metrics_data = data.get("metrics", {})
        spd_value = metrics_data.get("statistical_parity_difference", 0)
        di_value = metrics_data.get("disparate_impact", 0)
        max_group = metrics_data.get("max_positive_rate_group", "N/A")
        min_group = metrics_data.get("min_positive_rate_group", "N/A")

        attr_display = col_name.replace("_combined", "").replace("_", " + ")
        lines.append(f"**Fairness Metrics: {attr_display}**")
        lines.append("")
        lines.append(f"- **Stat Parity Diff:** {spd_value:.4f}")
        lines.append(f"- **Disparate Impact:** {di_value:.4f}")
        if max_group != "N/A" and min_group != "N/A":
            lines.append(f"- **Highest Rate Group:** {max_group}")
            lines.append(f"- **Lowest Rate Group:** {min_group}")

        sanitized_col = col_name.replace("_combined", "").replace(" + ", "_").replace(" ", "")
        file_prefix = f"{prefix}_" if prefix else ""
        csv_filename = f"{folder_name}/{file_prefix}fairness_stats_{sanitized_col}.csv"
        lines.append(f"- **Detailed CSV data:** `{csv_filename}`")
        lines.append("")


def format_ml_model_markdown(lines: List[str], ml_results: Dict[str, Any], title: str = "Machine Learning Evaluation Model") -> None:
    """Format ML model details into markdown (appends to *lines*)."""
    if not ml_results or ml_results.get("status") != "success":
        return
    model_type = ml_results.get("model_type")
    if not model_type:
        return
        
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"- **Algorithm:** {model_type}")
    
    test_size = ml_results.get("test_size")
    if test_size is not None:
        lines.append(f"- **Test Size:** {test_size}")
        
    accuracy = ml_results.get("performance", {}).get("accuracy")
    if accuracy is not None:
        lines.append(f"- **Accuracy:** {accuracy:.4f}")
    
    params = ml_results.get("model_params") or {}
    if params:
        params_str = ", ".join(f"`{k}={v}`" for k, v in params.items())
        lines.append(f"- **Parameters:** {params_str}")
    else:
        lines.append("- **Parameters:** Default settings")
    lines.append("")

    format_fairness_board_markdown(lines, ml_results, title)


# ── Report Generation ──────────────────────────────────────────────────

import os
import hashlib
from datetime import datetime
from pipeline.stages.base import safe_json_dumps

def generate_markdown_report(pipeline) -> str:
    """Generate pure markdown report (human-readable, easy PDF conversion)."""
    dataset_hash = hashlib.md5(pipeline.current_dataset.encode()).hexdigest()[:8]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines: List[str] = []
    lines.append("# Dataset Fairness Evaluation Report")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Dataset:** {pipeline.current_dataset}")
    lines.append(f"- **Timestamp:** {ts}")
    lines.append(f"- **Dataset Hash:** {dataset_hash}")
    if hasattr(pipeline, "target_column") and pipeline.target_column:
        lines.append(f"- **Target Column:** {pipeline.target_column}")
    lines.append(f"- **Objective:** {pipeline.user_objective or 'Dataset auditing'}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    stage_titles = {
        "0_loading": "Stage 0: Dataset Loading",
        "1_objective": "Stage 1: Objective Inspection",
        "2_quality": "Stage 2: Data Quality Analysis",
        "3_sensitive": "Stage 3: Sensitive Attribute Detection",
        "4_imbalance": "Stage 4: Imbalance Analysis",
        "4_5_target_fairness": "Stage 4.5: Target Fairness Analysis",
        "5_recommendations": "Stage 5: Recommendations",
        "6_bias_mitigation": "Stage 6: Bias Mitigation",
    }
    
    for stage_key, stage_data in pipeline.evaluation_results["stages"].items():
        title = stage_titles.get(stage_key, stage_key.replace("_", " ").title())
        lines.append(f"## {title}")
        lines.append("")
        
        if not isinstance(stage_data, dict):
            lines.append(str(stage_data))
            lines.append("")
            lines.append("---")
            lines.append("")
            continue
        
        if "tool_used" in stage_data:
            lines.append(f"**Tool Used:** `{stage_data['tool_used']}`")
            lines.append("")
        
        if stage_key == "6_bias_mitigation" and "methods" in stage_data:
            format_mitigation_markdown(lines, stage_data)
        elif "agent_analysis" in stage_data:
            lines.append("### Analysis")
            lines.append("")
            
            # Add Pair Selection before Target Fairness
            if stage_key == "4_5_target_fairness":
                sensitive_stage = pipeline.evaluation_results["stages"].get("3_sensitive", {})
                pair_sel = sensitive_stage.get("pair_selection")
                if pair_sel:
                    format_pair_selection_markdown(lines, pair_sel)
                    
            # Add ML Model info before analysis
            if "ml_model_results" in stage_data:
                format_ml_model_markdown(lines, stage_data["ml_model_results"], title="Base Fairness ML Model")
            if "intersectional_ml_results" in stage_data:
                format_ml_model_markdown(lines, stage_data["intersectional_ml_results"], title="Intersectional Fairness ML Model")
                
            lines.append(stage_data["agent_analysis"])
            lines.append("")
                
        elif "recommendations" in stage_data:
            lines.append("### Recommendations")
            lines.append("")
            lines.append(stage_data["recommendations"])
            lines.append("")
        elif "agent_response" in stage_data:
            lines.append("### Response")
            lines.append("")
            lines.append(str(stage_data["agent_response"]))
            lines.append("")
        else:
            if "objective" in stage_data:
                lines.append(f"**Objective:** {stage_data.get('objective', 'N/A')}")
                lines.append("")
            if "validation" in stage_data:
                lines.append(f"**Validation:** {stage_data.get('validation', 'N/A')}")
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    lines.append("*Report generated by Dataset Fairness Evaluation System*")
    return "\n".join(lines)


def generate_json_data(pipeline) -> Dict[str, Any]:
    """Generate JSON file with all tool results organized by stage."""
    dataset_hash = hashlib.md5(pipeline.current_dataset.encode()).hexdigest()[:8]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    json_data = {
        "metadata": {
            "dataset": pipeline.current_dataset,
            "timestamp": ts,
            "dataset_hash": dataset_hash,
            "target_column": getattr(pipeline, "target_column", None),
            "objective": pipeline.user_objective or "Dataset auditing",
            "report_directory": pipeline.report_dir,
        },
        "stages": {}
    }
    
    for stage_key, stage_data in pipeline.evaluation_results["stages"].items():
        stage_json = {}
        
        if isinstance(stage_data, dict):
            if "tool_used" in stage_data:
                stage_json["tool_used"] = stage_data["tool_used"]
            if "tool_result" in stage_data:
                stage_json["tool_result"] = stage_data["tool_result"]
            if "pair_selection" in stage_data:
                stage_json["pair_selection"] = stage_data["pair_selection"]
            if "ml_model_results" in stage_data:
                stage_json["ml_model_results"] = stage_data["ml_model_results"]
            if "intersectional_ml_results" in stage_data:
                stage_json["intersectional_ml_results"] = stage_data["intersectional_ml_results"]
            if "methods" in stage_data:
                stage_json["methods"] = stage_data["methods"]
                stage_json["applied_methods"] = stage_data.get("applied_methods", [])
                stage_json["status"] = stage_data.get("status", "unknown")
        else:
            stage_json["data"] = stage_data
        
        json_data["stages"][stage_key] = stage_json
    
    return json_data


def save_fairness_comparison_files(pipeline):
    """Save individual fairness comparison JSON files for each method."""
    stage_data = pipeline.evaluation_results["stages"].get("6_bias_mitigation", {})
    methods_results = stage_data.get("methods", {})
    
    for method, mr in methods_results.items():
        mitigation_result = mr.get("mitigation_result", {})
        fairness_comparison = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
        
        if fairness_comparison and fairness_comparison.get("status") != "error":
            try:
                fn = f"fairness_comparison_{method.lower().replace(' ', '_')}.json"
                fp = os.path.join(pipeline.report_dir, fn)
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(safe_json_dumps(fairness_comparison))
            except Exception as exc:
                print(f"Warning: Could not save fairness comparison JSON: {exc}")

    save_fairness_csv_files(pipeline)


def save_fairness_csv_files(pipeline):
    """Save fairness detailed group statistics as CSV files in the report directory."""
    import csv

    def extract_and_save(ml_results: Dict[str, Any], title: str):
        if not ml_results or ml_results.get("status") != "success":
            return
        fairness = ml_results.get("fairness_analysis", {})
        if not fairness:
            return
            
        folder_name, prefix = _get_csv_folder_and_prefix(title)
        
        target_dir = os.path.join(pipeline.report_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        for col_name, data in fairness.items():
            groups_data = data.get("groups", {})
            if not groups_data:
                continue
            
            sanitized_col = col_name.replace("_combined", "").replace(" + ", "_").replace(" ", "")
            file_prefix = f"{prefix}_" if prefix else ""
            csv_filename = f"{file_prefix}fairness_stats_{sanitized_col}.csv"
            csv_path = os.path.join(target_dir, csv_filename)
            
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Group", "Count", "Accuracy", "F1 Score", "Selection Rate", "Base Rate", "FNR", "FPR", "TPR", "TNR"])
                    
                    for group, metrics in groups_data.items():
                        g_name = group.replace("_", " + ")
                        
                        def fmt(v):
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                if isinstance(v, float):
                                    return f"{v:.4f}"
                                return str(v)
                            return str(v)

                        writer.writerow([
                            g_name,
                            fmt(metrics.get("count", "N/A")),
                            fmt(metrics.get("accuracy", "N/A")),
                            fmt(metrics.get("f1_macro", "N/A")),
                            fmt(metrics.get("positive_rate", "N/A")),
                            fmt(metrics.get("base_rate", "N/A")),
                            fmt(metrics.get("fnr", "N/A")),
                            fmt(metrics.get("fpr", "N/A")),
                            fmt(metrics.get("tpr", "N/A")),
                            fmt(metrics.get("tnr", "N/A"))
                        ])
            except Exception as exc:
                print(f"Warning: Could not save fairness CSV file {csv_filename}: {exc}")

    stages = pipeline.evaluation_results.get("stages", {})
    
    # Check stage 4 Base Model
    if "4_imbalance" in stages and "ml_model_results" in stages["4_imbalance"]:
        extract_and_save(stages["4_imbalance"]["ml_model_results"], "Base Fairness ML Model")
        
    # Check stage 4.5 Target Fairness
    if "4_5_target_fairness" in stages and "intersectional_ml_results" in stages["4_5_target_fairness"]:
        extract_and_save(stages["4_5_target_fairness"]["intersectional_ml_results"], "Intersectional Fairness ML Model")
        
    # Check stage 6 Bias Mitigation
    if "6_bias_mitigation" in stages:
        mr_dict = stages["6_bias_mitigation"].get("methods", {})
        for method, mr in mr_dict.items():
            mitigation_result = mr.get("mitigation_result", {})
            fc = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
            if fc:
                mitigated_metrics = fc.get("mitigated_metrics", {})
                if mitigated_metrics and mitigated_metrics.get("status") == "success":
                    extract_and_save(mitigated_metrics, f"Evaluation ML Model ({method})")
