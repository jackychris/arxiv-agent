# memory/long_term.py
import re

from db import memory as db_memory
from db import pool as db_pool

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
    "fetch_url": ["arxiv", "truncation", "truncated"],
    "get_paper_content": [
        "fetch the abstract directly via fetch_url",
        "fetch the abstract page directly",
        "fallback after fetch_url",
    ],
}


def _is_stale(tool: str, lesson: str) -> bool:
    text = lesson.lower()
    return any(p in text for p in _STALE_PATTERNS.get(tool, []))


def _extract_lesson(reflection: str | dict) -> str:
    if isinstance(reflection, dict):
        return str(reflection.get("lesson") or "").strip()
    return re.sub(r"\s+", " ", str(reflection).strip())


async def add(tool: str, reflection: str | dict) -> None:
    lesson = _extract_lesson(reflection)
    if not lesson or _is_stale(tool, lesson):
        return
    pool = await db_pool.get()
    await db_memory.add(pool, tool, lesson)


async def get(tool: str) -> list[str]:
    pool = await db_pool.get()
    return await db_memory.get(pool, tool)


async def get_all() -> dict[str, list[str]]:
    pool = await db_pool.get()
    return await db_memory.get_all(pool, _TOOLS)
