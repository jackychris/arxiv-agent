# graph/events.py
from __future__ import annotations

import json
from typing import Any

from graph.errors import AgentError
from schemas import SSEEvent

PLAN = "plan"
STEP = "step"
OBSERVATION = "observation"
TASK_RESULT = "task_result"
TASK_DONE = "task_done"
NODE_RESULT = "node_result"
SYNTHESIZING = "synthesizing"
CRITIQUE = "critique"
HUMAN_REVIEW = "human_review"
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


def plan(tasks: list[dict], *, run_id: str, round: int | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"tasks": tasks}
    if round is not None:
        data["round"] = round
    return event(PLAN, run_id=run_id, data=data, tasks=tasks)


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


def node_result(
    node: str,
    *,
    run_id: str,
    task_id: str | None = None,
    content: str | None = None,
    data: dict[str, Any] | None = None,
    status: str = "done",
) -> dict[str, Any]:
    return event(
        NODE_RESULT,
        run_id=run_id,
        task_id=task_id,
        content=content,
        data={"node": node, "status": status, **(data or {})},
    )


def synthesizing(*, run_id: str) -> dict[str, Any]:
    return event(SYNTHESIZING, run_id=run_id)


def critique(ok: bool, gaps: list[str], *, run_id: str) -> dict[str, Any]:
    content = "Answer is sufficient." if ok else f"{len(gaps)} gap(s) identified, launching follow-up research."
    return event(CRITIQUE, run_id=run_id, content=content, data={"ok": ok, "gaps": gaps})


def human_review(
    stage: str,
    prompt: str,
    options: list[dict[str, Any]],
    *,
    run_id: str | None,
    payload: dict[str, Any] | None = None,
    interrupt_id: str | None = None,
) -> dict[str, Any]:
    return event(
        HUMAN_REVIEW,
        run_id=run_id,
        content=prompt,
        data={
            "stage": stage,
            "options": options,
            "payload": payload or {},
            "interrupt_id": interrupt_id,
        },
    )


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
