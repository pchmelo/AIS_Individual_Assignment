from __future__ import annotations
from typing import Any, Dict

from pipeline.stages.base import BaseStageExecutor, safe_json_dumps


"""
Stage 2 – Data Quality Analysis.
Analyse missing data and other quality issues.
"""

class QualityStage(BaseStageExecutor):

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        tool_result = ctx["fairness_tools"].check_missing_data(ctx["dataset_name"])
        prompt = (
            "Analyze this missing data report and provide insights: "
            + safe_json_dumps(tool_result)
        )
        return self._tool_then_analyze(
            "check_missing_data", tool_result, prompt, stage,
        )
