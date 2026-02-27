"""Stage 6 – Bias Mitigation."""

from __future__ import annotations

import os
from typing import Any, Dict

from pipeline.stages.base import BaseStageExecutor, safe_json_dumps


class MitigationStage(BaseStageExecutor):
    """Apply bias-mitigation techniques and compare fairness metrics."""

    METHOD_MAP = {
        "Reweighting": "reweighting",
        "SMOTE": "smote",
        "Random Oversampling": "oversampling",
        "Random Undersampling": "undersampling",
    }

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        mitigation_cfg = ctx.get("mitigation_config")
        if not mitigation_cfg or not mitigation_cfg.get("methods"):
            return {"status": "skipped", "message": "No mitigation methods selected."}

        sensitive_cols: list = (
            ctx["results"].get("3_sensitive", {}).get("sensitive_columns", [])
        )

        all_results: Dict[str, Any] = {}
        selected_methods = mitigation_cfg["methods"]

        for method_name, method_params in selected_methods.items():
            try:
                method_key = self.METHOD_MAP.get(method_name, method_name.lower())
                sens = method_params.get("sensitive_columns", sensitive_cols)
                extra = {k: v for k, v in method_params.items() if k != "sensitive_columns"}

                mitigation_result = _apply_single_mitigation(
                    method=method_key,
                    ctx=ctx,
                    sensitive_columns=sens,
                    extra_params=extra,
                )

                if mitigation_result.get("status") == "success":
                    comparison_result = _compare_datasets(
                        ctx=ctx,
                        mitigated_dataset=mitigation_result.get("output_file"),
                        sensitive_columns=sensitive_cols,
                        agent=ctx["pipeline"].recommendation_agent,
                    )
                    all_results[method_name] = {
                        "status": "success",
                        "method_params": method_params,
                        "mitigation_result": mitigation_result,
                        "comparison_result": comparison_result,
                        "fairness_comparison": mitigation_result.get("fairness_comparison"),
                    }
                else:
                    all_results[method_name] = {
                        "status": "error",
                        "error": mitigation_result.get("message", "Unknown error"),
                    }
            except Exception as exc:
                all_results[method_name] = {"status": "error", "error": str(exc)}

        return {
            "status": "success",
            "methods": all_results,
            "applied_methods": list(selected_methods.keys()),
        }


# ── Module-level helpers (kept out of the class for reuse) ───────────


def _apply_single_mitigation(
    method: str,
    ctx: Dict[str, Any],
    sensitive_columns: list,
    extra_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply one bias-mitigation technique and optionally run a fairness comparison."""
    bias_tools = ctx["bias_mitigation_tools"]
    fairness_tools = ctx["fairness_tools"]
    dataset_name = ctx["dataset_name"]
    target_column = ctx["target_column"]

    output_dir = os.path.join(ctx["report_dir"], "generated_csv")
    os.makedirs(output_dir, exist_ok=True)

    shared = dict(dataset_name=dataset_name, target_column=target_column, output_dir=output_dir)

    if method == "reweighting":
        if not sensitive_columns:
            return {"status": "error", "message": "Sensitive columns required for reweighting"}
        result = bias_tools.apply_reweighting(sensitive_columns=sensitive_columns, **shared)
    elif method == "smote":
        result = bias_tools.apply_smote(
            k_neighbors=extra_params.get("k_neighbors", 5),
            sampling_strategy=extra_params.get("sampling_strategy", "auto"),
            **shared,
        )
    elif method == "oversampling":
        result = bias_tools.apply_oversampling(
            sampling_strategy=extra_params.get("sampling_strategy", "auto"),
            **shared,
        )
    elif method == "undersampling":
        result = bias_tools.apply_undersampling(
            sampling_strategy=extra_params.get("sampling_strategy", "auto"),
            **shared,
        )
    else:
        return {"status": "error", "message": f"Unknown method: {method}"}

    # ── Fairness comparison against baseline ─────────────────────────
    if result.get("status") == "success" and result.get("output_file"):
        baseline = ctx["results"].get("4_imbalance", {}).get("baseline_fairness_metrics")
        analyzed_cols = ctx["results"].get("4_imbalance", {}).get("analyzed_columns", [])

        if (
            baseline
            and baseline.get("status") == "success"
            and analyzed_cols
            and target_column
        ):
            mitigated_metrics = fairness_tools.train_and_evaluate_proxy_model(
                dataset_name=result["output_file"],
                target_column=target_column,
                sensitive_columns=analyzed_cols,
                test_size=0.25,
                model_type="Random Forest",
                model_params={},
            )
            if mitigated_metrics.get("status") == "success":
                result["fairness_comparison"] = _compare_fairness_metrics(
                    baseline, mitigated_metrics, method,
                )

    return result


def _compare_datasets(
    ctx: Dict[str, Any],
    mitigated_dataset: str,
    sensitive_columns: list,
    agent,
) -> Dict[str, Any]:
    """Compare original vs mitigated dataset and have the agent analyse."""
    result = ctx["bias_mitigation_tools"].compare_datasets(
        original_dataset=ctx["dataset_name"],
        mitigated_dataset=mitigated_dataset,
        target_column=ctx["target_column"],
        sensitive_columns=sensitive_columns,
    )

    prompt = (
        "Analyze the comparison between original and mitigated datasets:\n\n"
        f"{safe_json_dumps(result)}\n\n"
        "Provide a detailed analysis:\n"
        "1. Was the bias mitigation effective? (Yes/No and why)\n"
        "2. What improved? (specific metrics and percentages)\n"
        "3. What remained problematic? (if any)\n"
        "4. Recommendations for further improvements\n\n"
        "Be specific with numbers and provide actionable insights."
    )
    result["agent_analysis"] = agent.run(prompt)
    return result


def _compare_fairness_metrics(
    baseline: Dict[str, Any],
    mitigated: Dict[str, Any],
    method_name: str,
) -> Dict[str, Any]:
    """Compute per-attribute fairness improvement between baseline and mitigated."""
    if not baseline or baseline.get("status") != "success":
        return {"status": "error", "message": "Invalid baseline metrics"}
    if not mitigated or mitigated.get("status") != "success":
        return {"status": "error", "message": "Invalid mitigated metrics"}

    comparison: Dict[str, Any] = {
        "method": method_name,
        "baseline_metrics": baseline,
        "mitigated_metrics": mitigated,
        "improvements": {},
        "per_attribute_comparison": {},
    }

    baseline_fairness = baseline.get("fairness_analysis", {})
    mitigated_fairness = mitigated.get("fairness_analysis", {})

    for attr in baseline_fairness:
        if attr not in mitigated_fairness:
            continue

        b_metrics = baseline_fairness[attr].get("metrics", {})
        m_metrics = mitigated_fairness[attr].get("metrics", {})

        spd_b = b_metrics.get("statistical_parity_difference", 0)
        spd_m = m_metrics.get("statistical_parity_difference", 0)

        di_b = b_metrics.get("disparate_impact", 0)
        di_m = m_metrics.get("disparate_impact", 0)

        comparison["per_attribute_comparison"][attr] = {
            "statistical_parity_difference": {
                "baseline": float(spd_b),
                "mitigated": float(spd_m),
                "change": float(spd_b - spd_m),
                "improved": bool(abs(spd_m) < abs(spd_b)),
            },
            "disparate_impact": {
                "baseline": float(di_b),
                "mitigated": float(di_m),
                "change": float(di_m - di_b),
                "improved": bool(abs(1.0 - di_m) < abs(1.0 - di_b)),
            },
        }

    improvements_count = sum(
        1
        for ac in comparison["per_attribute_comparison"].values()
        if ac["statistical_parity_difference"]["improved"]
        or ac["disparate_impact"]["improved"]
    )
    total_metrics = len(comparison["per_attribute_comparison"]) * 2

    if total_metrics == 0:
        comparison["overall_improvement"] = "Unknown"
    elif improvements_count > total_metrics * 0.6:
        comparison["overall_improvement"] = "Significant"
    elif improvements_count > total_metrics * 0.3:
        comparison["overall_improvement"] = "Moderate"
    elif improvements_count > 0:
        comparison["overall_improvement"] = "Minor"
    else:
        comparison["overall_improvement"] = "None or Negative"

    return comparison
