from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    "synthesizing",
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


MemoryOutcome = Literal["success", "failure", "mixed"]


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    lesson: str
    outcome: MemoryOutcome = "mixed"
    source: str = "reflection"
    tags: list[str] = Field(default_factory=list)
    error_code: str | None = None
    created_at: str = Field(default_factory=utc_now)
    last_seen_at: str = Field(default_factory=utc_now)
    seen: int = Field(default=1, ge=1)

    @field_validator("lesson")
    @classmethod
    def _lesson_required(cls, lesson: str) -> str:
        lesson = lesson.strip()
        if not lesson:
            raise ValueError("lesson is required")
        return lesson

    @field_validator("tags")
    @classmethod
    def _compact_tags(cls, tags: list[str]) -> list[str]:
        compacted = []
        for tag in tags:
            tag = str(tag).strip()
            if tag and tag not in compacted:
                compacted.append(tag)
        return compacted[:6]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


TraceKind = Literal[
    "run_start",
    "plan_created",
    "task_start",
    "agent_step",
    "tool_result",
    "task_result",
    "task_done",
    "synthesis_start",
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
