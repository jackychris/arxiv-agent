# tools/arxiv/summarize.py
import json
import re
from urllib.parse import urlparse

import llm
from prompts import SUMMARIZE_PROMPT
from tools.web.web_tools import fetch_url

from .paper_detail import get_paper_detail

_ARXIV_ID_RE = re.compile(r"/(?:abs|pdf|html)/([^/?#]+)")


async def summarize_paper(paper_id: str) -> dict:
    detail = await get_paper_detail(paper_id)
    if "error" in detail:
        return detail

    content = detail.get("content", "")
    if not content:
        return {"error": "No content available to summarize"}

    prompt = SUMMARIZE_PROMPT.format(content=content)
    response = await llm.chat([{"role": "user", "content": prompt}])

    summary = json.loads(response)

    return {
        "id": detail["id"],
        "title": detail["title"],
        "authors": detail["authors"],
        "published": detail["published"],
        "url": detail["url"],
        **summary,
    }


async def get_paper_content(
    arxiv_id: str | None = None,
    pdf_url: str | None = None,
) -> dict:
    if arxiv_id:
        return await summarize_paper(arxiv_id)
    if pdf_url:
        host = (urlparse(pdf_url).hostname or "").lower()
        if host == "arxiv.org" or host.endswith(".arxiv.org"):
            match = _ARXIV_ID_RE.search(pdf_url)
            if match:
                return await summarize_paper(match.group(1).removesuffix(".pdf"))
        return await fetch_url(pdf_url)
    return {"error": "Either arxiv_id or pdf_url must be provided"}
