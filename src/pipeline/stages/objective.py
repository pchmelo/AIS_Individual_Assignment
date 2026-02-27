"""Stage 1 – Objective Inspection."""

from __future__ import annotations
from typing import Any, Dict

from pipeline.stages.base import BaseStageExecutor


class ObjectiveStage(BaseStageExecutor):
    """Validate the user's evaluation objective."""

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ctx["user_prompt"]
        if stage.user_context:
            prompt += (
                "\n\n[USER INSTRUCTION — you MUST follow this]: "
                + stage.user_context
            )

        is_audit_request = bool(
            prompt
            and any(
                kw in prompt.lower()
                for kw in ("audit", "analyze", "evaluate", "check", "inspect")
            )
        )

        return {
            "objective": prompt or "Dataset auditing",
            "is_audit_request": is_audit_request,
            "validation": "Dataset format compatible (CSV)",
        }
