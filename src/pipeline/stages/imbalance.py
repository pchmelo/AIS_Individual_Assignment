"""Stage 4 – Imbalance Analysis."""

from __future__ import annotations
from typing import Any, Dict

from pipeline.stages.base import BaseStageExecutor, safe_json_dumps


class ImbalanceStage(BaseStageExecutor):
    """Measure class imbalance for sensitive columns, with optional proxy model."""

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        results = ctx["results"]

        # Apply user-confirmed sensitive columns if provided
        confirmed = ctx.get("confirmed_sensitive_columns")
        if confirmed:
            results.setdefault("3_sensitive", {})["sensitive_columns"] = confirmed

        sensitive_cols: list = (
            results.get("3_sensitive", {}).get("sensitive_columns", [])
        )

        # ── Tool call ────────────────────────────────────────────────
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

        # ── Optional proxy model ─────────────────────────────────────
        proxy_config = ctx.get("proxy_config", {})
        target = ctx.get("target_column")
        proxy_results = None

        if (
            proxy_config.get("enabled")
            and sensitive_cols
            and target
        ):
            proxy_results = ctx["fairness_tools"].train_and_evaluate_proxy_model(
                dataset_name=ctx["dataset_name"],
                target_column=target,
                sensitive_columns=sensitive_cols,
                test_size=proxy_config.get("test_size", 0.25),
                model_type=proxy_config.get("model_type", "Random Forest"),
                model_params=proxy_config.get("model_params", {}),
            )

        # ── Agent analysis ───────────────────────────────────────────
        proxy_context = _build_proxy_context(proxy_results)

        prompt = (
            "Analyze class imbalance in SENSITIVE/PROTECTED attributes ONLY.\n\n"
            f"SENSITIVE COLUMNS IDENTIFIED: {', '.join(sensitive_cols)}\n\n"
            "IMBALANCE DATA (for sensitive columns only):\n"
            f"{safe_json_dumps(tool_result)}\n"
            f"{proxy_context}\n\n"
            "Provide:\n"
            "1. Summary of imbalance severity for each sensitive column\n"
            "2. Fairness risks (which groups are underrepresented?)\n"
            "3. Impact on model bias\n"
            "4. Specific mitigation recommendations\n\n"
            "Focus ONLY on the sensitive columns listed above."
        )

        return self._tool_then_analyze(
            "check_class_imbalance",
            tool_result,
            prompt,
            stage,
            proxy_model_results=proxy_results,
            baseline_fairness_metrics=proxy_results,
            analyzed_columns=sensitive_cols,
        )


# ── Module-level helper (reused by fairness.py) ─────────────────────────


def _build_proxy_context(proxy_results: dict | None) -> str:
    """Format proxy-model results into a prompt snippet."""
    if not proxy_results or proxy_results.get("status") != "success":
        return ""

    per_label_str = ""
    if "per_label_metrics" in proxy_results.get("performance", {}):
        per_label_str = (
            "\nPer-Label Performance (F1, Precision, Recall):\n"
            + safe_json_dumps(proxy_results["performance"]["per_label_metrics"])
        )

    return (
        "PROXY MODEL FAIRNESS ANALYSIS:\n"
        f"Model: {proxy_results.get('model_type')} "
        f"(Acc: {proxy_results['performance']['accuracy']}, "
        f"F1: {proxy_results['performance']['f1_macro']})\n"
        f"{per_label_str}\n\n"
        "Fairness Metrics per Attribute (F1 Score & Disparity):\n"
        f"{safe_json_dumps(proxy_results.get('fairness_analysis', {}))}\n\n"
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
