"""
Pipeline configuration – loads stage definitions from YAML.

:func:`load_pipeline_config` reads ``pipeline_config.yml`` (or a custom path),
resolves executor class names to instances from :mod:`pipeline.stages`, and
returns a list of :class:`StageDefinition` dataclasses that the pipeline
iterates to build concrete :class:`Stage` objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Type

import yaml

from pipeline.stages import (
    BaseStageExecutor,
    LoadingStage,
    ObjectiveStage,
    QualityStage,
    SensitiveDetectionStage,
    ImbalanceStage,
    TargetFairnessStage,
    RecommendationsStage,
    MitigationStage,
)

# ── Executor class registry ──────────────────────────────────────────────

_EXECUTOR_REGISTRY: Dict[str, Type[BaseStageExecutor]] = {
    "LoadingStage": LoadingStage,
    "ObjectiveStage": ObjectiveStage,
    "QualityStage": QualityStage,
    "SensitiveDetectionStage": SensitiveDetectionStage,
    "ImbalanceStage": ImbalanceStage,
    "TargetFairnessStage": TargetFairnessStage,
    "RecommendationsStage": RecommendationsStage,
    "MitigationStage": MitigationStage,
}


# ── Data class ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageDefinition:
    """Declarative description of a single pipeline stage."""

    key: str
    name: str
    executor: BaseStageExecutor
    agent_attr: str                    # attribute name on the pipeline
    description: str = ""
    requires_confirmation: bool = False
    optional: bool = False
    requires_target: bool = False      # only include when a target column is set


# ── Loader ───────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pipeline_config.yml")


def load_pipeline_config(path: str | None = None) -> List[StageDefinition]:
    """Load stage definitions from a YAML file.

    Args:
        path: Path to the YAML file.  Defaults to
              ``pipeline/pipeline_config.yml`` next to this module.

    Returns:
        Ordered list of :class:`StageDefinition` instances.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If an executor class name is not recognised.
    """
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


# Convenience: pre-loaded default config (same as before for backwards compat)
EVALUATION_STAGES: List[StageDefinition] = load_pipeline_config()
