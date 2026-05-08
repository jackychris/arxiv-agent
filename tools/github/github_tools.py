# tools/github/github_tools.py
import base64
import logging

import httpx

from config import GITHUB_HTTP_TIMEOUT, GITHUB_TOKEN
from utils import retry_async

from ._rate_limit import github_api_call

_BASE = "https://api.github.com"
_RETRY_DELAYS = [10.0, 30.0]

logger = logging.getLogger(__name__)


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


async def _get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    async def _do() -> httpx.Response:
        r = await github_api_call(client.get(url, headers=_headers(), **kwargs))
        if r.status_code == 429 or r.status_code >= 500:
            logger.warning("GitHub %d, will retry", r.status_code)
            r.raise_for_status()
        return r

    return await retry_async(
        _do,
        retry_delays=_RETRY_DELAYS,
        is_retryable_exception=_is_retryable,
        exhausted_exception=lambda exc, _: exc or RuntimeError("GitHub retries exhausted"),
    )


async def search_repos(
    query: str,
    language: str | None = None,
    sort: str = "stars",
    max_results: int = 5,
) -> list[dict]:
    max_results = min(max_results, 10)
    q = f"{query} language:{language}" if language else query
    async with httpx.AsyncClient(timeout=GITHUB_HTTP_TIMEOUT) as client:
        r = await _get(
            client,
            f"{_BASE}/search/repositories",
            params={"q": q, "sort": sort, "order": "desc", "per_page": max_results},
        )
        r.raise_for_status()
    return [
        {
            "full_name": item["full_name"],
            "description": item["description"],
            "url": item["html_url"],
            "stars": item["stargazers_count"],
            "forks": item["forks_count"],
            "language": item["language"],
            "topics": item["topics"],
            "updated": item["updated_at"][:10],
        }
        for item in r.json()["items"]
    ]


async def get_repo_readme(full_name: str) -> dict:
    async with httpx.AsyncClient(timeout=GITHUB_HTTP_TIMEOUT) as client:
        r = await _get(client, f"{_BASE}/repos/{full_name}/readme")
        if r.status_code == 404:
            return {"error": f"No README found for {full_name}"}
        r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    if len(content) > 8000:
        content = content[:8000] + "\n\n[truncated]"
    return {"full_name": full_name, "readme": content}


async def search_code(
    query: str,
    language: str | None = None,
    max_results: int = 5,
) -> list[dict]:
    max_results = min(max_results, 10)
    q = f"{query} language:{language}" if language else query
    async with httpx.AsyncClient(timeout=GITHUB_HTTP_TIMEOUT) as client:
        r = await _get(
            client,
            f"{_BASE}/search/code",
            params={"q": q, "per_page": max_results},
        )
        r.raise_for_status()
    return [
        {
            "repo": item["repository"]["full_name"],
            "path": item["path"],
            "url": item["html_url"],
        }
        for item in r.json()["items"]
    ]
