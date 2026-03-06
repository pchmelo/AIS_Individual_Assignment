from __future__ import annotations
from typing import Any, Dict

from pipeline.stages.base import BaseStageExecutor, safe_json_dumps


"""
Stage 4 – Imbalance Analysis.
Measure class imbalance for sensitive columns, with optional ml model.
"""

class ImbalanceStage(BaseStageExecutor):

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        results = ctx["results"]

        # Apply user-confirmed sensitive columns if provided
        confirmed = ctx.get("confirmed_sensitive_columns")
        if confirmed:
            results.setdefault("3_sensitive", {})["sensitive_columns"] = confirmed

        sensitive_cols: list = (
            results.get("3_sensitive", {}).get("sensitive_columns", [])
        )

        # Tool call
        tool_result = ctx["fairness_tools"].check_class_imbalance(
            ctx["dataset_name"],
        )

        # Keep only the sensitive columns in the result
        if tool_result.get("status") == "success" and sensitive_cols:
            filtered = [
                d for d in tool_result.get("details", [])
                if d["column"] in sensitive_cols
            ]
            tool_result["details"] = filtered
            tool_result["imbalanced_columns"] = len(filtered)

        # Optional ml model
        ml_config = ctx.get("ml_config", {})
        target = ctx.get("target_column")
        ml_results = None

        if (
            ml_config.get("enabled")
            and sensitive_cols
            and target
        ):
            ml_results = ctx["fairness_tools"].train_and_evaluate_ml_model(
                dataset_name=ctx["dataset_name"],
                target_column=target,
                sensitive_columns=sensitive_cols,
                test_size=ml_config.get("test_size", 0.25),
                model_type=ml_config.get("model_type", "Random Forest"),
                model_params=ml_config.get("model_params", {}),
            )

        # Agent analysis
        ml_context = _build_ml_context(ml_results)

        prompt = (
            "Analyze class imbalance in SENSITIVE/PROTECTED attributes ONLY.\n\n"
            f"SENSITIVE COLUMNS IDENTIFIED: {', '.join(sensitive_cols)}\n\n"
            "IMBALANCE DATA (for sensitive columns only):\n"
            f"{safe_json_dumps(tool_result)}\n"
            f"{ml_context}\n\n"
            "Provide:\n"
            "1. Summary of imbalance severity for each sensitive column\n"
            "2. Fairness risks (which groups are underrepresented?)\n"
            "3. Impact on model bias\n"
            "4. Specific mitigation recommendations\n\n"
            "FORMATTING RULES:\n"
            "- Use ## for main headers, ### for subsections\n"
            "- Use numbered lists (1. 2. 3.) for ordered items\n"
            "- Use bullet points (- item) for unordered lists\n"
            "- Do NOT use ** bold markers ** around headers\n"
            "- Do NOT use emojis, icons, or special symbols (no ✓, ✗, ■, ●, etc.)\n\n"
            "Focus ONLY on the sensitive columns listed above."
        )

        return self._tool_then_analyze(
            "check_class_imbalance",
            tool_result,
            prompt,
            stage,
            ml_model_results=ml_results,
            baseline_fairness_metrics=ml_results,
            analyzed_columns=sensitive_cols,
        )


def _build_ml_context(ml_results: dict | None) -> str:
    """Format ML model results into a prompt snippet."""
    if not ml_results or ml_results.get("status") != "success":
        return ""

    per_label_str = ""
    if "per_label_metrics" in ml_results.get("performance", {}):
        per_label_str = (
            "\nPer-Label Performance (F1, Precision, Recall):\n"
            + safe_json_dumps(ml_results["performance"]["per_label_metrics"])
        )

    return (
        "ML MODEL FAIRNESS ANALYSIS:\n"
        f"Model: {ml_results.get('model_type')} "
        f"(Acc: {ml_results['performance']['accuracy']}, "
        f"F1: {ml_results['performance']['f1_macro']})\n"
        f"{per_label_str}\n\n"
        "Fairness Metrics per Attribute (F1 Score & Disparity):\n"
        f"{safe_json_dumps(ml_results.get('fairness_analysis', {}))}\n\n"
        "Include these metrics (Statistical Parity, Disparate Impact, "
        "Group F1, FNR/FPR Ratios) in your assessment.\n\n"
        "CRITICAL ANALYSIS REQUIREMENTS:\n"
        '1. Compare "Base Rate" (Actual % Positive) vs "Selection Rate" '
        "(Predicted % Positive) for each group.\n"
        "2. High FNR (False Negative Rate) in a protected group means the "
        "model fails to select qualified candidates from that group. "
        "Highlight this.\n"
        '3. Calculate and mention the "FNR Ratio" (Max FNR / Min FNR) if '
        "significant disparity exists.\n"
        "4. Identify if the model *amplifies* existing bias "
        "(e.g. if Selection Rate disparity > Base Rate disparity)."
    )
