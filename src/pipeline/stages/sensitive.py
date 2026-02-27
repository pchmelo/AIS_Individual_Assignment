"""Stage 3 – Sensitive Attribute Detection."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from pipeline.stages.base import BaseStageExecutor


class SensitiveDetectionStage(BaseStageExecutor):
    """Detect protected / sensitive columns using an LLM."""

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        columns_result = ctx["fairness_tools"].detect_sensitive_attributes(
            ctx["dataset_name"],
        )

        simplified = self._create_simplified_column_summary(
            columns_result.get("columns", []),
        )

        target_column = ctx.get("target_column")
        target_note = ""
        if target_column:
            target_note = (
                f"\n\nIMPORTANT: EXCLUDE the target column '{target_column}' "
                "from sensitive attributes – it is the variable being "
                "predicted, not a protected attribute."
            )

        prompt = (
            "Analyze this dataset and identify ALL SENSITIVE/PROTECTED "
            "attribute columns.\n\n"
            "KEY SENSITIVE ATTRIBUTES TO LOOK FOR:\n"
            "- Demographics: Age, Race, Ethnicity, Sex/Gender\n"
            "- Personal: Religion, Marital-status, Relationship\n"
            "- Socioeconomic: Income, Education, Occupation\n"
            "- Geographic: Native-country, Nationality\n\n"
            f"{simplified}{target_note}\n\n"
            "IMPORTANT: Look at BOTH column names AND their "
            "values/distributions:\n"
            "- Race column with values like White, Black, Asian → SENSITIVE\n"
            "- Sex column with Male/Female → SENSITIVE\n"
            "- Native-country with country names → SENSITIVE\n"
            "- Age with numeric ages → SENSITIVE\n"
            "- Education levels → SENSITIVE\n"
            "- Marital-status → SENSITIVE\n"
            "- Income/salary → SENSITIVE\n\n"
            "For EACH sensitive column, output EXACTLY this format:\n"
            "Column: [exact_column_name] | Reason: [why_sensitive] "
            "| Values: [key_values]\n\n"
            "List ALL sensitive columns – don't miss Race, Sex, "
            "Native-country if present."
        )
        prompt = self._append_user_context(prompt, stage.user_context)

        analysis = stage.agent.run(prompt, max_tokens=4096)

        # Extract column names from the agent response
        identified: List[str] = list(
            dict.fromkeys(re.findall(r"Column:\s*([\w-]+)", analysis))
        )

        if target_column and target_column in identified:
            identified.remove(target_column)

        return {
            "tool_used": "detect_sensitive_attributes",
            "tool_result": columns_result,
            "simplified_summary": simplified,
            "agent_analysis": analysis,
            "sensitive_columns": identified,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_simplified_column_summary(columns_data: List[Dict]) -> str:
        lines = [
            "COLUMN SUMMARY TABLE:",
            "=" * 100,
            f"{'Column':<25} {'Type':<10} {'Unique':<8} "
            f"{'Sample Values / Top Categories':<50}",
            "=" * 100,
        ]
        for col in columns_data:
            name = col["column"]
            dtype = col["type"]
            unique = col["unique_values"]
            if "top_values" in col:
                top_items = list(col["top_values"].items())[:3]
                values_str = ", ".join(f"{k}({v}%)" for k, v in top_items)
            else:
                values_str = str(col["sample_values"][:5])
            lines.append(f"{name:<25} {dtype:<10} {unique:<8} {values_str:<50}")
        return "\n".join(lines)
