from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    message: str
    source: str | None = None
    recoverable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    tool: str
    data: Any = None
    error: ErrorEnvelope | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _error_matches_ok(self):
        if self.ok is False and self.error is None:
            raise ValueError("error is required when ok=false")
        if self.ok is True and self.error is not None:
            raise ValueError("error must be null when ok=true")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


SSEType = Literal[
    "plan",
    "step",
    "observation",
    "task_result",
    "task_done",
    "node_result",
    "synthesizing",
    "critique",
    "human_review",
    "final",
    "error",
]


class SSEEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: SSEType
    ts: str = Field(default_factory=utc_now)
    run_id: str | None = None
    task_id: str | None = None
    content: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: ErrorEnvelope | dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


TraceKind = Literal[
    "run_start",
    "query_rewritten",
    "plan_created",
    "graph_task_start",
    "agent_task_start",
    "agent_step",
    "tool_result",
    "graph_task_result",
    "agent_task_result",
    "graph_task_done",
    "agent_task_done",
    "synthesis_start",
    "critique",
    "run_final",
    "run_error",
]


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: TraceKind
    ts: str = Field(default_factory=utc_now)
    run_id: str
    task_id: str | None = None
    step: int | None = None
    tool: str | None = None
    tool_call_id: str | None = None
    latency_ms: int | None = None
    ok: bool | None = None
    error_code: str | None = None
    content: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
