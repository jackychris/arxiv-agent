# memory/long_term.py
import asyncio
import json
import os
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from schemas import MemoryEntry

_PATH = os.path.join(os.path.dirname(__file__), "long_term.json")
_LOCK = asyncio.Lock()
_MAX_ACTIVE_PER_TOOL = 6
_MAX_STORED_PER_TOOL = 30

_TOOLS = [
    "search_semantic_scholar",
    "search_arxiv",
    "get_paper_content",
    "search_repos",
    "get_repo_readme",
    "search_code",
    "web_search",
    "fetch_url",
    "orchestrator",
]

_STALE_PATTERNS = {
    "fetch_url": [
        "arxiv",
        "truncation",
        "truncated",
    ],
    "summarize_paper": [
        "fetch the abstract directly via fetch_url",
        "fetch the abstract page directly",
        "fallback after fetch_url",
    ],
}


def _load() -> dict:
    if os.path.exists(_PATH):
        with open(_PATH) as f:
            data = json.load(f)
    else:
        data = {}
    for tool in _TOOLS:
        data.setdefault(tool, [])
    return data


def _save(data: dict) -> None:
    with open(_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_lesson(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _entry_lesson(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("lesson") or "")
    return str(entry)


def _normalize_entry(tool: str, reflection: str | dict) -> dict:
    now = _now()
    if isinstance(reflection, dict):
        lesson = str(reflection.get("lesson") or "").strip()
        tags = reflection.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        entry = {
            "tool": tool,
            "lesson": lesson,
            "outcome": reflection.get("outcome") or "mixed",
            "source": reflection.get("source") or "reflection",
            "tags": tags[:6],
            "error_code": reflection.get("error_code"),
            "created_at": reflection.get("created_at") or now,
            "last_seen_at": reflection.get("last_seen_at") or now,
            "seen": int(reflection.get("seen") or 1),
        }
    else:
        entry = {
            "tool": tool,
            "lesson": str(reflection).strip(),
            "outcome": "mixed",
            "source": "legacy",
            "tags": [],
            "error_code": None,
            "created_at": now,
            "last_seen_at": now,
            "seen": 1,
        }
    try:
        return MemoryEntry(**entry).to_dict()  # type: ignore[arg-type]
    except ValidationError:
        fallback: dict[str, object] = {
            "tool": tool,
            "lesson": str(entry.get("lesson") or "").strip() or "Unspecified lesson.",
            "outcome": "mixed",
            "source": str(entry.get("source") or "reflection"),
            "tags": [],
            "error_code": None,
            "created_at": now,
            "last_seen_at": now,
            "seen": 1,
        }
        return MemoryEntry(**fallback).to_dict()  # type: ignore[arg-type]


def _merge_entries(entries: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for entry in entries:
        lesson_key = _normalize_lesson(entry.get("lesson", ""))
        if not lesson_key:
            continue
        existing = merged.get(lesson_key)
        if existing is None:
            merged[lesson_key] = entry
            continue
        existing["seen"] = int(existing.get("seen") or 1) + int(entry.get("seen") or 1)
        existing["last_seen_at"] = max(
            str(existing.get("last_seen_at") or ""), str(entry.get("last_seen_at") or "")
        )
        existing_tags = list(existing.get("tags") or [])
        for tag in entry.get("tags") or []:
            if tag not in existing_tags:
                existing_tags.append(tag)
        existing["tags"] = existing_tags[:6]
        if not existing.get("error_code") and entry.get("error_code"):
            existing["error_code"] = entry["error_code"]
    return list(merged.values())


def _rank_entries(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda e: (int(e.get("seen") or 1), str(e.get("last_seen_at") or "")),
        reverse=True,
    )


def _compact_entries(tool: str, entries: list[Any], *, active_only: bool) -> list[dict]:
    normalized = [_normalize_entry(tool, entry) for entry in entries]
    active = [entry for entry in normalized if not _is_stale(tool, entry)]
    ranked = _rank_entries(_merge_entries(active))
    limit = _MAX_ACTIVE_PER_TOOL if active_only else _MAX_STORED_PER_TOOL
    return ranked[:limit]


async def add(tool: str, reflection: str | dict) -> None:
    async with _LOCK:
        data = _load()
        if tool not in data:
            data[tool] = []
        entries = [_normalize_entry(tool, entry) for entry in data[tool]]
        new_entry = _normalize_entry(tool, reflection)
        if not new_entry["lesson"] or _is_stale(tool, new_entry):
            return
        entries.append(new_entry)
        data[tool] = _compact_entries(tool, entries, active_only=False)
        _save(data)


def _is_stale(tool: str, reflection: Any) -> bool:
    text = _entry_lesson(reflection).lower()
    return any(pattern in text for pattern in _STALE_PATTERNS.get(tool, []))


def _active_entries(tool: str, entries: list[Any]) -> list[dict]:
    return _compact_entries(tool, entries, active_only=True)


def get(tool: str) -> list[dict]:
    return _active_entries(tool, _load().get(tool, []))


def get_all() -> dict:
    return {tool: _active_entries(tool, entries) for tool, entries in _load().items()}
