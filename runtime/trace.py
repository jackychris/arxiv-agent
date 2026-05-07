import json
import time
from pathlib import Path
from typing import Any

from config import ENABLE_TRACE, TRACE_DIR
from runtime.storage import JsonlTraceStore, TraceStore
from schemas import TraceEvent

_store: TraceStore = JsonlTraceStore(TRACE_DIR)


def monotonic_ms() -> int:
    return int(time.perf_counter() * 1000)


def elapsed_ms(start_ms: int) -> int:
    return max(0, monotonic_ms() - start_ms)


def set_store(store: TraceStore) -> None:
    global _store
    _store = store


def get_store() -> TraceStore:
    return _store


def trace_path(run_id: str) -> Path:
    path = _store.path(run_id)
    if path is None:
        raise RuntimeError("current trace store does not expose file paths")
    return path


def record(
    kind: str,
    *,
    run_id: str,
    task_id: str | None = None,
    step: int | None = None,
    tool: str | None = None,
    tool_call_id: str | None = None,
    latency_ms: int | None = None,
    ok: bool | None = None,
    error_code: str | None = None,
    content: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = TraceEvent(
        kind=kind,  # type: ignore[arg-type]
        run_id=run_id,
        task_id=task_id,
        step=step,
        tool=tool,
        tool_call_id=tool_call_id,
        latency_ms=latency_ms,
        ok=ok,
        error_code=error_code,
        content=content,
        data=data or {},
    ).to_dict()
    if ENABLE_TRACE:
        _store.append(run_id, event)
    return event


def summarize_tool_result(observation: str) -> tuple[bool | None, str | None]:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    ok = payload.get("ok")
    error_code = None
    error = payload.get("error")
    if isinstance(error, dict):
        error_code = error.get("code")
    return ok if isinstance(ok, bool) else None, error_code
