# tools/arxiv/semantic_scholar.py
import asyncio

import httpx

from config import SEMANTIC_SCHOLAR_API_KEY
from utils import retry_async

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,authors,year,citationCount,externalIds,tldr"
_LOCK = asyncio.Lock()
_RETRY_DELAYS = [10.0, 30.0]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


async def _get(path: str, params: dict) -> dict:
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    async with _LOCK:
        if not SEMANTIC_SCHOLAR_API_KEY:
            await asyncio.sleep(1.1)

        async def _do() -> dict:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(f"{_BASE}{path}", params=params, headers=headers)
                r.raise_for_status()
                return r.json()

        return await retry_async(
            _do,
            retry_delays=_RETRY_DELAYS,
            is_retryable_exception=_is_retryable,
            exhausted_exception=lambda exc, _: (
                exc or RuntimeError("Semantic Scholar retries exhausted")
            ),
        )


async def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    year_from: int | None = None,
) -> list[dict]:
    max_results = min(max_results, 10)
    params: dict = {"query": query, "limit": max_results, "fields": _FIELDS}
    if year_from:
        params["year"] = f"{year_from}-"

    data = await _get("/paper/search", params)

    results = []
    for p in data.get("data", []):
        arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
        tldr = (p.get("tldr") or {}).get("text")
        results.append(
            {
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "authors": [a["name"] for a in (p.get("authors") or [])[:3]],
                "year": p.get("year"),
                "citation_count": p.get("citationCount"),
                "arxiv_id": arxiv_id,
                "tldr": tldr,
            }
        )
    return results
