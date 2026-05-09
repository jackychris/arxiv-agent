# tools/arxiv/summarize.py
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import llm
from config import PAPER_CONTENT_DIR
from db import papers as db_papers
from db import pool as db_pool
from prompts import SUMMARIZE_PROMPT
from tools.web.web_tools import fetch_url

from .paper_detail import get_paper_detail

_ARXIV_ID_RE = re.compile(r"/(?:abs|pdf|html)/([^/?#]+)")
_CONTENT_DIR = Path(PAPER_CONTENT_DIR)


def _content_path(arxiv_id: str) -> Path:
    safe = arxiv_id.replace("/", "_")
    prefix = safe[:4] if len(safe) >= 4 else "misc"
    return _CONTENT_DIR / prefix / f"{safe}.txt"


def _row_to_result(row: dict) -> dict:
    return {
        "id": row["arxiv_id"],
        "title": row["title"],
        "authors": row["authors"],
        "published": row["published"],
        "url": row["url"],
        "summary": row["summary"],
        "keywords": row["keywords"],
    }


async def summarize_paper(paper_id: str) -> dict:
    paper_id = re.sub(r"v\d+$", "", paper_id)  # strip version suffix
    pool = await db_pool.get()
    row = await db_papers.get(pool, paper_id)

    # Level 1: full summary already in DB
    if row and row["has_summary"]:
        return _row_to_result(row)

    # Level 2: content file exists + meta in DB — skip arxiv fetch, just run LLM
    content_file = _content_path(paper_id)
    if content_file.exists() and row:
        content = content_file.read_text()
    else:
        # Level 3: fetch from arxiv
        detail = await get_paper_detail(paper_id)
        if "error" in detail:
            return detail
        content = detail.get("content", "")
        if not content:
            return {"error": "No content available to summarize"}
        content_file.parent.mkdir(parents=True, exist_ok=True)
        content_file.write_text(content)
        detail["id"] = paper_id  # ensure consistent ID (no version suffix)
        await db_papers.upsert_meta(pool, detail)
        await db_papers.mark_content(pool, paper_id)
        row = await db_papers.get(pool, paper_id)

    # Run LLM summarize
    prompt = SUMMARIZE_PROMPT.format(content=content)
    response = await llm.chat([{"role": "user", "content": prompt}])
    summary = json.loads(response)

    await db_papers.save_summary(pool, paper_id, summary)

    return {
        "id": paper_id,
        "title": row["title"] if row else "",
        "authors": row["authors"] if row else [],
        "published": row["published"] if row else "",
        "url": row["url"] if row else "",
        "summary": summary.get("summary", ""),
        "keywords": summary.get("keywords", []),
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
