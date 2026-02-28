from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np



def convert_to_serializable(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to native Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(e) for e in obj]
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    return obj


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """JSON-serialize *data*, converting numpy types first."""
    try:
        return json.dumps(convert_to_serializable(data), indent=indent)
    except Exception as exc:
        return f"Error serializing data: {exc}"


# Base executor

class BaseStageExecutor(ABC):
    """
    Abstract base for every stage executor.
    """

    @abstractmethod
    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Run the stage logic and return a result dict."""
        pass

    @staticmethod
    def _append_user_context(prompt: str, user_context: str | None) -> str:
        """Append an optional user instruction to *prompt*."""
        if user_context:
            return prompt + (
                "\n\n[USER INSTRUCTION — you MUST follow this]: "
                + user_context
            )
        return prompt

    @staticmethod
    def _tool_then_analyze(
        tool_name: str,
        tool_result: Any,
        prompt: str,
        stage,
        **extra_fields: Any,
    ) -> Dict[str, Any]:
        """Common pattern shared by most stages.

        1. Append the user instruction (if any) to *prompt*.
        2. Ask ``stage.agent`` to analyse.
        3. Return a standard result dict.
        """
        if stage.user_context:
            prompt += (
                "\n\n[USER INSTRUCTION — you MUST follow this]: "
                + stage.user_context
            )
        analysis = stage.agent.run(prompt)
        result: Dict[str, Any] = {
            "tool_used": tool_name,
            "tool_result": tool_result,
            "agent_analysis": analysis,
        }
        result.update(extra_fields)
        return result
