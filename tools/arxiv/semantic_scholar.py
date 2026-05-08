# tools/arxiv/semantic_scholar.py
import asyncio

import httpx

from utils import retry_async

_BASE = "https://api.openalex.org"
_LOCK = asyncio.Lock()
_RETRY_DELAYS = [10.0, 30.0]
_SELECT = "id,title,abstract_inverted_index,authorships,publication_year,cited_by_count,ids,best_oa_location,primary_location,type"

# Venues considered top-tier for ML/CS/AI research
_TOP_VENUES = {
    "neurips",
    "icml",
    "iclr",
    "cvpr",
    "iccv",
    "eccv",
    "acl",
    "emnlp",
    "naacl",
    "aaai",
    "ijcai",
    "sigkdd",
    "sigir",
    "www",
    "usenix",
    "sosp",
    "osdi",
    "nature",
    "science",
    "cell",
    "lancet",
    "nejm",
    "ieee transactions",
    "acm transactions",
}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def _source_quality(work_type: str | None, source: dict) -> str:
    source_type = source.get("type", "")
    venue = (source.get("display_name") or "").lower()
    if work_type == "preprint":
        return "preprint"
    if any(top in venue for top in _TOP_VENUES):
        return "top-venue"
    if source_type in ("journal", "conference"):
        return "peer-reviewed"
    if source_type == "repository":
        return "preprint"
    return "unknown"


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    if not inverted_index:
        return None
    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


async def _get(path: str, params: dict) -> dict:
    async with _LOCK:
        await asyncio.sleep(0.15)  # ~7 req/s, well within 10 req/s free limit

        async def _do() -> dict:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(f"{_BASE}{path}", params=params)
                r.raise_for_status()
                return r.json()

        return await retry_async(
            _do,
            retry_delays=_RETRY_DELAYS,
            is_retryable_exception=_is_retryable,
            exhausted_exception=lambda exc, _: exc or RuntimeError("OpenAlex retries exhausted"),
        )


async def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    year_from: int | None = None,
) -> list[dict]:
    max_results = min(max_results, 10)
    params: dict = {
        "search": query,
        "per_page": max_results,
        "select": _SELECT,
    }
    if year_from:
        params["filter"] = f"publication_year:>{year_from - 1}"

    data = await _get("/works", params)

    results = []
    for p in data.get("results", []):
        arxiv_id = (p.get("ids") or {}).get("arxiv", "")
        if arxiv_id:
            arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "")
        authors = [
            a["author"]["display_name"] for a in (p.get("authorships") or [])[:3] if a.get("author")
        ]
        oa = p.get("best_oa_location") or {}
        pdf_url = oa.get("pdf_url")
        primary = p.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name")
        quality = _source_quality(p.get("type"), source)
        results.append(
            {
                "title": p.get("title"),
                "abstract": _reconstruct_abstract(p.get("abstract_inverted_index")),
                "authors": authors,
                "year": p.get("publication_year"),
                "citation_count": p.get("cited_by_count"),
                "arxiv_id": arxiv_id or None,
                "pdf_url": pdf_url,
                "venue": venue,
                "source_quality": quality,
            }
        )
    return results
