from __future__ import annotations
from typing import Any, Dict

from equiaudit.pipeline.stages.base import BaseStageExecutor, safe_json_dumps


"""
Stage 5 – Recommendations.
Generate improvement recommendations from compiled findings.
"""

class RecommendationsStage(BaseStageExecutor):

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        findings = self._compile_findings(ctx["results"])

        prompt = (
            f"Based on evaluation results for {ctx['dataset_name']}, provide:\n"
            "1. Top 3 critical issues\n"
            "2. Mitigation strategies (SMOTE, reweighting, etc.)\n"
            "3. Priority order\n"
            "4. Expected impact\n\n"
            "FORMATTING RULES (MUST FOLLOW):\n"
            "- Use ## for main section headers (e.g., ## Top 3 Critical Issues)\n"
            "- Use ### for subsection headers (e.g., ### 1. Severe Class Imbalance)\n"
            "- Use numbered lists (1. 2. 3.) for ordered items\n"
            "- Use bullet points (- item) for unordered lists\n"
            "- Do NOT use ** bold markers ** around section titles\n"
            "- Keep text clean without excessive formatting\n"
            "- Do NOT use emojis, icons, or special symbols (no ✓, ✗, ■, ●, etc.)\n\n"
            f"Findings: {findings}"
        )
        prompt = self._append_user_context(prompt, stage.user_context)

        recommendations = stage.agent.run(prompt)
        return {"recommendations": recommendations}

    @staticmethod
    def _compile_findings(stage_results: Dict[str, Any]) -> str:
        """Merge all stage outputs into a single summary string."""
        parts: list[str] = []
        for name, data in stage_results.items():
            parts.append(f"\n{name.upper()}:")
            parts.append(safe_json_dumps(data))
        return "\n".join(parts)
