# tools/web/web_tools.py
import asyncio
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx
from tavily import TavilyClient

from config import TAVILY_API_KEY, WEB_FETCH_TIMEOUT

from ._rate_limit import tavily_api_call

_ARXIV_ID_RE = re.compile(r"/(?:abs|pdf|html)/([^/?#]+)")

_HIGH_CREDIBILITY_DOMAINS = {
    "arxiv.org",
    "semanticscholar.org",
    "openalex.org",
    "github.com",
    "huggingface.co",
    "openai.com",
    "anthropic.com",
    "deepmind.com",
    "research.google.com",
    "ai.meta.com",
    "microsoft.com",
    "nvidia.com",
    "nature.com",
    "science.org",
    "acm.org",
    "ieee.org",
    "springer.com",
    "wikipedia.org",
    "readthedocs.io",
    "pytorch.org",
    "tensorflow.org",
}
_HIGH_CREDIBILITY_SUFFIXES = (".edu", ".gov")


def _credibility(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if host in _HIGH_CREDIBILITY_DOMAINS or host.endswith(_HIGH_CREDIBILITY_SUFFIXES):
        return "high"
    if any(host.endswith("." + d) for d in _HIGH_CREDIBILITY_DOMAINS):
        return "high"
    return "low"


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; arxiv-agent/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "tr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)
            self._parts.append(" ")

    def text(self) -> str:
        raw = unescape("".join(self._parts))
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def _get_client() -> TavilyClient:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=TAVILY_API_KEY)


async def web_search(query: str, max_results: int = 5, search_depth: str = "basic") -> list[dict]:
    max_results = min(max_results, 10)

    async def _do():
        def _fetch():
            client = _get_client()
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )
            return [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "content": r["content"],
                    "score": round(r.get("score", 0), 3),
                    "credibility": _credibility(r["url"]),
                }
                for r in response.get("results", [])
            ]

        return await asyncio.to_thread(_fetch)

    return await tavily_api_call(_do())


async def fetch_url(url: str) -> dict:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        match = _ARXIV_ID_RE.search(parsed.path)
        hint = " Use summarize_paper with the arXiv paper_id instead."
        if match:
            paper_id = match.group(1).removesuffix(".pdf")
            hint = f' Use summarize_paper({{"paper_id": "{paper_id}"}}) instead.'
        return {
            "url": url,
            "error": f"fetch_url does not support arXiv URLs.{hint}",
        }

    try:
        return await _fetch_http(url)
    except Exception as e:
        return {
            "url": url,
            "error": f"fetch_url failed after {WEB_FETCH_TIMEOUT:g}s",
            "details": {
                "exception_type": type(e).__name__,
                "message": str(e),
            },
        }


async def _fetch_http(url: str) -> dict:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(WEB_FETCH_TIMEOUT),
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type or "application/xhtml" in content_type:
            content = _html_to_text(response.text)
        else:
            content = response.text
        return {
            "url": str(response.url),
            "content": content,
            "content_type": content_type,
            "fetcher": "httpx",
        }


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()
