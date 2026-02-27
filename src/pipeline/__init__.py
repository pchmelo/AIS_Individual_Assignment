"""
pipeline – Dataset evaluation pipeline package.

Exports the core building blocks:

* :class:`Stage`, :class:`StageStatus`, :class:`NavigationAction`
* :class:`DatasetEvaluationPipeline`
"""

from pipeline.stage import Stage, StageStatus, NavigationAction
from pipeline.pipeline import DatasetEvaluationPipeline

__all__ = [
    "Stage",
    "StageStatus",
    "NavigationAction",
    "DatasetEvaluationPipeline",
]
