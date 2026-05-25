# memory/long_term.py
import re

from db import memory as db_memory
from db import pool as db_pool

_TOOLS = [
    "search_semantic",
    "search_arxiv",
    "search_openalex",
    "search_crossref",
    "get_crossref_paper_by_doi",
    "search_dblp",
    "download_arxiv",
    "download_semantic",
    "read_arxiv_paper",
    "read_semantic_paper",
    "read_openalex_paper",
    "read_dblp_paper",
    "search_repositories",
    "search_code",
    "get_file_contents",
    "tavily_search",
    "tavily_extract",
    "orchestrator",
]

_STALE_PATTERNS = {
    "tavily_extract": ["arxiv", "truncation", "truncated"],
    "read_arxiv_paper": [
        "fetch the abstract directly via tavily_extract",
        "fetch the abstract page directly",
        "fallback after tavily_extract",
    ],
    "read_semantic_paper": [
        "fetch the abstract directly via tavily_extract",
        "fetch the abstract page directly",
        "fallback after tavily_extract",
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
    if pool is None:
        return
    await db_memory.add(pool, tool, lesson)


async def get(tool: str) -> list[str]:
    pool = await db_pool.get()
    if pool is None:
        return []
    return await db_memory.get(pool, tool)


async def get_all() -> dict[str, list[str]]:
    pool = await db_pool.get()
    if pool is None:
        return {}
    return await db_memory.get_all(pool, _TOOLS)
