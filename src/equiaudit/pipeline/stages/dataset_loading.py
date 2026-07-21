from __future__ import annotations
from typing import Any, Dict

from pipeline.stages.base import BaseStageExecutor


"""
Stage 0 – Dataset Loading.
Load and validate the dataset file.
"""

class LoadingStage(BaseStageExecutor):

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        tool_result = ctx["fairness_tools"].load_dataset(ctx["dataset_name"])
        prompt = (
            f"The dataset '{ctx['dataset_name']}' has been loaded with the "
            f"following information: {tool_result}. "
            "Provide a brief summary of the dataset."
        )
        return self._tool_then_analyze("load_dataset", tool_result, prompt, stage)
