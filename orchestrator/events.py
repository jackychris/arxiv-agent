# orchestrator/events.py
from __future__ import annotations

import json
from typing import Any

from orchestrator.errors import AgentError
from schemas import SSEEvent

PLAN = "plan"
STEP = "step"
OBSERVATION = "observation"
TASK_RESULT = "task_result"
TASK_DONE = "task_done"
SYNTHESIZING = "synthesizing"
FINAL = "final"
ERROR = "error"


def event(
    event_type: str,
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    content: str | None = None,
    data: dict[str, Any] | None = None,
    error: AgentError | dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event_type,
        "run_id": run_id,
        "task_id": task_id,
        "content": content,
        "data": data or {},
        "error": error.to_dict() if isinstance(error, AgentError) else error,
    }
    payload.update(extra)
    return SSEEvent(**payload).to_dict()


def plan(tasks: list[dict], *, run_id: str) -> dict[str, Any]:
    return event(PLAN, run_id=run_id, data={"tasks": tasks}, tasks=tasks)


def step(
    task_id: str, content: str, *, run_id: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    return event(STEP, run_id=run_id, task_id=task_id, content=content, data=data)


def observation(
    task_id: str, content: str, *, run_id: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    return event(OBSERVATION, run_id=run_id, task_id=task_id, content=content, data=data)


def status_text(task_id: str, content: str, *, run_id: str) -> dict[str, Any]:
    structured = _parse_status_payload(content)
    kind = structured["kind"]
    visible_content = structured.get("content") or ""
    data = {k: v for k, v in structured.items() if k not in {"kind", "content"}}
    if kind == STEP:
        return step(task_id, visible_content, run_id=run_id, data=data)
    return observation(task_id, visible_content, run_id=run_id, data=data)


def task_result(
    task_id: str,
    content: str,
    *,
    run_id: str,
    error: AgentError | None = None,
) -> dict[str, Any]:
    return event(TASK_RESULT, run_id=run_id, task_id=task_id, content=content, error=error)


def task_done(task_id: str, *, run_id: str) -> dict[str, Any]:
    return event(TASK_DONE, run_id=run_id, task_id=task_id)


def synthesizing(*, run_id: str) -> dict[str, Any]:
    return event(SYNTHESIZING, run_id=run_id)


def final(content: str, *, run_id: str) -> dict[str, Any]:
    return event(FINAL, run_id=run_id, content=content)


def error(
    content: str, *, run_id: str | None = None, error: AgentError | None = None
) -> dict[str, Any]:
    return event(ERROR, run_id=run_id, content=content, error=error)


def encode_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def heartbeat() -> str:
    return ": heartbeat\n\n"


def _parse_status_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("A2A status message must be JSON") from e
    if not isinstance(payload, dict):
        raise ValueError("A2A status message must be a JSON object")
    if payload.get("kind") not in {STEP, OBSERVATION}:
        raise ValueError("A2A status message kind must be 'step' or 'observation'")
    return payload
