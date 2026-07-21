from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Type

import yaml

from equiaudit.pipeline.stages import (
    BaseStageExecutor,
    LoadingStage,
    ObjectiveStage,
    QualityStage,
    SensitiveDetectionStage,
    DiscretizationStage,
    ImbalanceStage,
    TargetFairnessStage,
    RecommendationsStage,
    MitigationStage,
)


_EXECUTOR_REGISTRY: Dict[str, Type[BaseStageExecutor]] = {
    "LoadingStage": LoadingStage,
    "ObjectiveStage": ObjectiveStage,
    "QualityStage": QualityStage,
    "SensitiveDetectionStage": SensitiveDetectionStage,
    "DiscretizationStage": DiscretizationStage,
    "ImbalanceStage": ImbalanceStage,
    "TargetFairnessStage": TargetFairnessStage,
    "RecommendationsStage": RecommendationsStage,
    "MitigationStage": MitigationStage,
}


@dataclass(frozen=True)
class StageDefinition:
    key: str
    name: str
    executor: BaseStageExecutor
    agent_attr: str                    # attribute name on the pipeline
    description: str = ""
    requires_confirmation: bool = False
    optional: bool = False
    requires_target: bool = False      # only include when a target column is set



_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pipeline_config.yml")


def load_pipeline_config(path: str | None = None) -> List[StageDefinition]:
    path = path or _DEFAULT_CONFIG_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    definitions: List[StageDefinition] = []

    for entry in raw.get("stages", []):
        executor_name = entry["executor"]
        executor_cls = _EXECUTOR_REGISTRY.get(executor_name)
        if executor_cls is None:
            raise ValueError(
                f"Unknown executor '{executor_name}'. "
                f"Available: {', '.join(_EXECUTOR_REGISTRY)}"
            )

        definitions.append(
            StageDefinition(
                key=entry["key"],
                name=entry["name"],
                executor=executor_cls(),
                agent_attr=entry["agent"],
                description=entry.get("description", ""),
                requires_confirmation=entry.get("requires_confirmation", False),
                optional=entry.get("optional", False),
                requires_target=entry.get("requires_target", False),
            )
        )

    return definitions


EVALUATION_STAGES: List[StageDefinition] = load_pipeline_config()
