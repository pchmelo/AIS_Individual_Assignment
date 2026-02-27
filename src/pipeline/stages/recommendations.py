"""Stage 5 – Recommendations."""

from __future__ import annotations
from typing import Any, Dict

from pipeline.stages.base import BaseStageExecutor, safe_json_dumps


class RecommendationsStage(BaseStageExecutor):
    """Generate improvement recommendations from compiled findings."""

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        findings = self._compile_findings(ctx["results"])

        prompt = (
            f"Based on evaluation results for {ctx['dataset_name']}, provide:\n"
            "1. Top 3 critical issues\n"
            "2. Mitigation strategies (SMOTE, reweighting, etc.)\n"
            "3. Priority order\n"
            "4. Expected impact\n\n"
            f"Findings: {findings}"
        )
        prompt = self._append_user_context(prompt, stage.user_context)

        recommendations = stage.agent.run(prompt)
        return {"recommendations": recommendations}

    # ------------------------------------------------------------------

    @staticmethod
    def _compile_findings(stage_results: Dict[str, Any]) -> str:
        """Merge all stage outputs into a single summary string."""
        parts: list[str] = []
        for name, data in stage_results.items():
            parts.append(f"\n{name.upper()}:")
            parts.append(safe_json_dumps(data))
        return "\n".join(parts)
