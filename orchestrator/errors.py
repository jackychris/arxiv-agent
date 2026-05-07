# orchestrator/errors.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentError:
    code: str
    message: str
    source: str
    recoverable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "recoverable": self.recoverable,
            "details": self.details,
        }

    def render(self) -> str:
        if self.code == "NO_RESULT":
            return "[no result]"
        if self.details.get("state"):
            return f"[{self.details['state']}: {self.message}]"
        if self.details.get("exception_type"):
            return f"[failed: {self.details['exception_type']}: {self.message}]"
        return f"[{self.code.lower()}: {self.message}]"


def no_result(source: str = "orchestrator") -> AgentError:
    return AgentError(
        code="NO_RESULT",
        message="task ended before producing a result",
        source=source,
        recoverable=True,
    )


def exception_error(exc: BaseException, source: str, code: str = "INTERNAL_ERROR") -> AgentError:
    message = str(exc) or type(exc).__name__
    return AgentError(
        code=code,
        message=message,
        source=source,
        recoverable=True,
        details={"exception_type": type(exc).__name__},
    )


def task_state_error(state_name: str, reason: str, source: str = "a2a") -> AgentError:
    code = f"A2A_TASK_{state_name.upper()}"
    return AgentError(
        code=code,
        message=reason,
        source=source,
        recoverable=state_name in {"failed", "canceled"},
        details={"state": state_name},
    )


def failed_text(error: AgentError) -> str:
    return error.render()


def is_failure_text(text: str) -> bool:
    return (
        text.startswith("[failed") or text.startswith("[failed:") or text.startswith("[canceled:")
    )
