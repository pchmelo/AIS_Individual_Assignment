from __future__ import annotations
from typing import Any, Dict

from equiaudit.pipeline.stages.base import BaseStageExecutor, safe_json_dumps


"""
Stage 2 – Data Quality Analysis.
Analyse missing data and other quality issues.
"""

class QualityStage(BaseStageExecutor):

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        tool_result = ctx["fairness_tools"].check_missing_data(ctx["dataset_name"])
        prompt = (
            "Analyze this missing data report and provide insights.\n\n"
            f"DATA: {safe_json_dumps(tool_result)}\n\n"
            "FORMATTING RULES:\n"
            "- Use ## for main headers, ### for subsections\n"
            "- Use numbered lists (1. 2. 3.) for ordered items\n"
            "- Use bullet points (- item) for unordered lists\n"
            "- Do NOT use ** bold markers ** around headers\n"
            "- Do NOT use emojis, icons, or special symbols (no ✓, ✗, ■, ●, etc.)\n"
        )
        return self._tool_then_analyze(
            "check_missing_data", tool_result, prompt, stage,
        )
