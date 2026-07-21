from __future__ import annotations
import os
from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
from equiaudit.pipeline.stages.base import BaseStageExecutor, safe_json_dumps



"""
Stage 4.5 – Target Fairness Analysis.
Analyse fairness of the target variable across sensitive groups.
"""

class TargetFairnessStage(BaseStageExecutor):

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        results = ctx["results"]
        target_column = ctx["target_column"]
        sensitive_cols: List[str] = list(
            results.get("3_sensitive", {}).get("sensitive_columns", [])
        )

        # Exclude target from sensitive list
        if target_column in sensitive_cols:
            sensitive_cols = [c for c in sensitive_cols if c != target_column]

        if not sensitive_cols:
            return {
                "status": "skipped",
                "message": "No sensitive columns identified for fairness analysis",
            }

        selected_pairs = ctx.get("selected_pairs")

        # Tool call
        tool_result = ctx["fairness_tools"].analyze_target_fairness(
            dataset_name=ctx["dataset_name"],
            target_column=target_column,
            sensitive_columns=sensitive_cols,
            output_dir=ctx["images_dir"],
            selected_pairs=selected_pairs,
        )

        # Optional ML models
        ml_config = ctx.get("ml_config", {})
        single_attribute_ml = None
        intersectional_ml = None

        if ml_config.get("enabled"):
            single_attribute_ml = ctx["fairness_tools"].train_and_evaluate_ml_model(
                dataset_name=ctx["dataset_name"],
                target_column=target_column,
                sensitive_columns=sensitive_cols,
                test_size=ml_config.get("test_size", 0.25),
                model_type=ml_config.get("model_type", "Random Forest"),
                model_params=ml_config.get("model_params", {}),
            )

        if (
            ml_config.get("enabled")
            and selected_pairs
        ):
            intersectional_ml = self._run_intersectional_ml(
                ctx, selected_pairs, target_column, ml_config,
            )

        # Agent analysis
        ml_context = ""
        if single_attribute_ml and single_attribute_ml.get("status") == "success":
            ml_context += (
                "SINGLE-ATTRIBUTE ML MODEL METRICS:\n"
                f"(Model: {single_attribute_ml.get('model_type')})\n\n"
                "Fairness Metrics per Individual Sensitive Attribute:\n"
                f"{safe_json_dumps(single_attribute_ml.get('fairness_analysis', {}))}\n\n"
            )
        if intersectional_ml and intersectional_ml.get("status") == "success":
            ml_context += (
                "INTERSECTIONAL ML MODEL METRICS:\n"
                f"(Model: {intersectional_ml.get('model_type')})\n\n"
                "Fairness Metrics for Combined Groups (Intersectional):\n"
                f"{safe_json_dumps(intersectional_ml.get('fairness_analysis', {}))}\n\n"
                "CRITICAL ANALYSIS REQUIREMENTS:\n"
                '1. Analyze "F1 Score" for each intersectional group. '
                "Identify which SPECIFIC combination (e.g. Black Female) "
                "has the lowest performance.\n"
                '2. Compare "Base Rate" vs "Selection Rate".\n'
                "3. Highlight FNR Disparities. Are certain combinations "
                "being systematically rejected (High FNR)?"
            )

        prompt = (
            f"Analyze the target fairness metrics for '{target_column}' "
            "across sensitive attributes.\n\n"
            f"SENSITIVE COLUMNS ANALYZED: {', '.join(sensitive_cols)}\n\n"
            f"FAIRNESS METRICS DATA:\n{safe_json_dumps(tool_result)}\n"
            f"{ml_context}\n\n"
            "Provide analysis on:\n"
            "1. Target distribution across different demographic groups\n"
            "2. Disparate impact – which groups have significantly "
            "different target rates?\n"
            "3. Intersectional fairness – combined effects of multiple "
            "sensitive attributes\n"
            "4. Statistical parity violations\n"
            "5. Risk of discrimination or bias in predictions\n\n"
            "CRITICAL INSTRUCTION: Do NOT provide any mitigation recommendations. Your sole job is to diagnose the biases. Recommendations will be generated in a later stage.\n\n"
            "FORMATTING RULES:\n"
            "- Use ## for main headers, ### for subsections\n"
            "- Use numbered lists (1. 2. 3.) for ordered items\n"
            "- Use bullet points (- item) for unordered lists\n"
            "- Do NOT use ** bold markers ** around headers\n"
            "- Do NOT use emojis, icons, or special symbols (no ✓, ✗, ■, ●, etc.)\n\n"
            "Focus on quantitative disparities and their implications."
        )

        return self._tool_then_analyze(
            "analyze_target_fairness",
            tool_result,
            prompt,
            stage,
            single_attribute_ml_results=single_attribute_ml,
            intersectional_ml_results=intersectional_ml,
            target_column=target_column,
            analyzed_sensitive_columns=sensitive_cols,
        )


    @staticmethod
    def _run_intersectional_ml(
        ctx: Dict[str, Any],
        selected_pairs: List,
        target_column: str,
        ml_config: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Train a ML model on combined attribute columns."""
        try:
            fairness_tools = ctx["fairness_tools"]
            path = fairness_tools._resolve_path(ctx["dataset_name"])
            df = pd.read_csv(path)

            temp_cols: List[str] = []
            for pair in selected_pairs:
                col1, col2 = pair[0], pair[1]
                if col1 in df.columns and col2 in df.columns:
                    combined = f"{col1}_{col2}_combined"
                    df[combined] = df[col1].astype(str) + "_" + df[col2].astype(str)
                    temp_cols.append(combined)

            if not temp_cols:
                return None

            ts = datetime.now().strftime("%H%M%S")
            temp_filename = f"temp_intersectional_{ts}.csv"
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
            )
            temp_path = os.path.join(data_dir, temp_filename)
            df.to_csv(temp_path, index=False)

            result = fairness_tools.train_and_evaluate_ml_model(
                dataset_name=temp_filename.replace(".csv", ""),
                target_column=target_column,
                sensitive_columns=temp_cols,
                test_size=ml_config.get("test_size", 0.25),
                model_type=ml_config.get("model_type", "Random Forest"),
                model_params=ml_config.get("model_params", {}),
            )

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return result
        except Exception as exc:
            print(f"Intersectional ML analysis failed: {exc}")
            return None
