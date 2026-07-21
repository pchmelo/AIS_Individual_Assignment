from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class NavigationAction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    REPEAT = "repeat"


@dataclass
class Stage:
    """A single pipeline stage.

    Attributes:
        key:            Unique identifier (e.g. ``"0_loading"``).
        name:           Human-readable display name.
        execute_fn:     Callable ``(stage, pipeline_ctx) -> result_dict``.
                        Receives this Stage instance (so it can read
                        ``user_context``) and a shared pipeline context dict.
        agent:          The agent object associated with this stage (may be ``None``).
        description:    Short description shown in the UI.
        status:         Current execution status.
        result:         The result dictionary produced by ``execute_fn``.
        user_context:   Optional extra text supplied by the user before
                        running / repeating this stage.
        history:        List of ``(user_context, result)`` pairs – one entry per
                        execution (including repeats).
        optional:       If ``True`` the stage may be skipped when a prerequisite
                        such as ``target_column`` is missing.
        requires_confirmation:
                        If ``True`` the pipeline pauses *before* this stage
                        so the user can provide input.
    """

    key: str
    name: str
    execute_fn: Optional[Callable[["Stage", Dict[str, Any]], Dict[str, Any]]] = None
    agent: Any = None
    description: str = ""
    status: StageStatus = StageStatus.NOT_STARTED
    result: Optional[Dict[str, Any]] = None
    user_context: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    optional: bool = False
    requires_confirmation: bool = False


    def execute(self, pipeline_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Run the stage and store the result."""
        if self.execute_fn is None:
            self.status = StageStatus.SKIPPED
            self.result = {"status": "skipped", "message": f"No execution function for {self.key}"}
            return self.result

        self.status = StageStatus.RUNNING
        try:
            result = self.execute_fn(self, pipeline_ctx)
            self.result = result
            self.status = StageStatus.COMPLETED
            self.history.append({
                "user_context": self.user_context,
                "result": copy.deepcopy(result),
            })
            return result
        except Exception as exc:
            self.status = StageStatus.ERROR
            self.result = {"status": "error", "message": str(exc)}
            self.history.append({
                "user_context": self.user_context,
                "result": self.result,
            })
            raise

    def reset(self):
        """Reset status and result so the stage can be re-executed."""
        self.status = StageStatus.NOT_STARTED
        self.result = None
        self.user_context = None


    @property
    def is_completed(self) -> bool:
        return self.status == StageStatus.COMPLETED

    @property
    def is_skipped(self) -> bool:
        return self.status == StageStatus.SKIPPED

    def __repr__(self) -> str:
        return f"Stage(key={self.key!r}, name={self.name!r}, status={self.status.value})"
