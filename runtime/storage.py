from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

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
    def publish_citations(self, context_id: str, task_id: str, citations: list[dict]) -> None: ...

    def publish_history(self, context_id: str, task_id: str, history: str) -> None: ...

    def get_citations_from_others(self, context_id: str, task_id: str) -> list[dict]: ...

    def get_citations(self, context_id: str) -> list[dict]: ...

    def get_histories(self, context_id: str) -> list[dict]: ...

    def clear(self, context_id: str) -> None: ...


class InMemoryRunContextStore:
    def __init__(self):
        # context_id -> {key: citation_dict}
        self._citations: dict[str, dict[str, dict]] = {}
        # context_id -> task_id -> set of keys
        self._by_task: dict[str, dict[str, set[str]]] = {}
        # context_id -> next citation ref number
        self._next_ref_num: dict[str, int] = {}
        # context_id -> [{task_id, history}]
        self._histories: dict[str, list[dict]] = {}

    def publish_citations(self, context_id: str, task_id: str, citations: list[dict]) -> None:
        ctx = self._citations.setdefault(context_id, {})
        task_keys = self._by_task.setdefault(context_id, {}).setdefault(task_id, set())
        next_ref_num = self._next_ref_num.setdefault(context_id, 1)
        for c in citations:
            key = c.get("key")
            if key:
                existing = ctx.get(key)
                merged = dict(existing or {})
                merged.update(c)
                merged["key"] = key
                merged["cited"] = bool((existing or {}).get("cited")) or bool(c.get("cited"))

                existing_title = str((existing or {}).get("title") or "").strip()
                incoming_title = str(c.get("title") or "").strip()
                if len(existing_title) >= len(incoming_title):
                    merged["title"] = existing_title
                else:
                    merged["title"] = incoming_title

                if existing and existing.get("ref_num") is not None:
                    merged["ref_num"] = existing["ref_num"]
                else:
                    merged.setdefault("ref_num", next_ref_num)
                    next_ref_num = max(next_ref_num, int(merged["ref_num"]) + 1)
                ctx[key] = merged
                task_keys.add(key)
        self._next_ref_num[context_id] = next_ref_num

    def publish_history(self, context_id: str, task_id: str, history: str) -> None:
        if history:
            self._histories.setdefault(context_id, []).append(
                {"task_id": task_id, "history": history}
            )

    def get_citations_from_others(self, context_id: str, task_id: str) -> list[dict]:
        ctx = self._citations.get(context_id, {})
        my_keys = self._by_task.get(context_id, {}).get(task_id, set())
        return [c for key, c in ctx.items() if key not in my_keys]

    def get_citations(self, context_id: str) -> list[dict]:
        citations = list(self._citations.get(context_id, {}).values())
        return sorted(citations, key=lambda c: (c.get("ref_num") is None, c.get("ref_num", 0), c.get("key", "")))

    def get_histories(self, context_id: str) -> list[dict]:
        return list(self._histories.get(context_id, []))

    def clear(self, context_id: str) -> None:
        self._citations.pop(context_id, None)
        self._by_task.pop(context_id, None)
        self._next_ref_num.pop(context_id, None)
        self._histories.pop(context_id, None)
