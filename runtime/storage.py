from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from config import RUN_CONTEXT_ENTRY_MAX_CHARS

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


def safe_id(value: str) -> str:
    return _SAFE_ID.sub("_", value)[:120] or "unknown"


@contextmanager
def exclusive_file_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "posix":
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class TraceStore(Protocol):
    def append(self, run_id: str, event: dict) -> None: ...

    def path(self, run_id: str) -> Path | None: ...


class JsonlTraceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, run_id: str) -> Path:
        return self.root / f"{safe_id(run_id)}.jsonl"

    def append(self, run_id: str, event: dict) -> None:
        path = self.path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with exclusive_file_lock(path):
            with path.open("a", encoding="utf-8") as f:
                f.write(line)


class RunContextStore(Protocol):
    def publish(self, context_id: str, task_id: str, step: int, content: str) -> None: ...

    def get_entries(self, context_id: str) -> list[tuple[str, int, str]]: ...

    def clear(self, context_id: str) -> None: ...


class InMemoryRunContextStore:
    def __init__(self):
        # context_id -> list of (task_id, step, content)
        self._entries: dict[str, list[tuple[str, int, str]]] = {}

    def publish(self, context_id: str, task_id: str, step: int, content: str) -> None:
        entries = self._entries.setdefault(context_id, [])
        entries.append((task_id, step, compact_run_context_content(content)))

    def get_entries(self, context_id: str) -> list[tuple[str, int, str]]:
        return list(self._entries.get(context_id, []))

    def clear(self, context_id: str) -> None:
        self._entries.pop(context_id, None)


def compact_run_context_content(content: str, max_chars: int = RUN_CONTEXT_ENTRY_MAX_CHARS) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return _limit(content, max_chars)

    if not isinstance(payload, dict) or "ok" not in payload or "tool" not in payload:
        return _limit(content, max_chars)

    tool = payload.get("tool") or "unknown"
    ok = payload.get("ok")
    parts = [f"tool={tool}", f"ok={str(ok).lower()}"]
    if ok is False:
        error = payload.get("error") or {}
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            if code:
                parts.append(f"error_code={code}")
            if message:
                parts.append(f"error={_limit(str(message), 160)}")
        return _limit(" ".join(parts), max_chars)

    preview = _data_preview(payload.get("data"))
    if preview:
        parts.append(f"preview={preview}")
    return _limit(" ".join(parts), max_chars)


def _data_preview(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, list):
        items = []
        for item in data[:3]:
            if isinstance(item, dict):
                items.append(
                    str(item.get("title") or item.get("id") or item.get("url") or item)[:120]
                )
            else:
                items.append(str(item)[:120])
        return "; ".join(items)
    if isinstance(data, dict):
        if isinstance(data.get("content"), str):
            return _limit(data["content"], 220)
        for key in ("title", "summary", "url", "id"):
            if data.get(key):
                return _limit(str(data[key]), 220)
        return _limit(json.dumps(data, ensure_ascii=False), 220)
    return _limit(str(data), 220)


def _limit(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
