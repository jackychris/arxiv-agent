# tools/arxiv/summarize.py
import json
import logging
import re
import time
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
logger = logging.getLogger(__name__)


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
    started = time.monotonic()
    paper_id = re.sub(r"v\d+$", "", paper_id)  # strip version suffix
    logger.info("summarize_paper start: %s", paper_id)
    pool = await db_pool.get()
    row = await db_papers.get(pool, paper_id) if pool else None

    # Level 1: full summary already in DB
    if row and row["has_summary"]:
        logger.info(
            "summarize_paper cache hit summary: %s total=%.2fs",
            paper_id,
            time.monotonic() - started,
        )
        return _row_to_result(row)

    # Level 2: content file exists + meta in DB — skip arxiv fetch, just run LLM
    content_file = _content_path(paper_id)
    if content_file.exists() and row:
        logger.info("summarize_paper content file hit: %s", content_file)
        content = content_file.read_text()
    else:
        # Level 3: fetch from arxiv
        logger.info("summarize_paper fetching detail: %s", paper_id)
        detail = await get_paper_detail(paper_id)
        if "error" in detail:
            logger.info(
                "summarize_paper detail error: %s error=%s total=%.2fs",
                paper_id,
                detail.get("error"),
                time.monotonic() - started,
            )
            return detail
        content = detail.get("content", "")
        if not content:
            logger.info(
                "summarize_paper no content: %s total=%.2fs",
                paper_id,
                time.monotonic() - started,
            )
            return {"error": "No content available to summarize"}
        content_file.parent.mkdir(parents=True, exist_ok=True)
        content_file.write_text(content)
        if pool:
            detail["id"] = paper_id
            await db_papers.upsert_meta(pool, detail)
            await db_papers.mark_content(pool, paper_id)
            row = await db_papers.get(pool, paper_id)

    # Run LLM summarize
    logger.info("summarize_paper llm start: %s", paper_id)
    prompt = SUMMARIZE_PROMPT.format(content=content)
    response = await llm.chat([{"role": "user", "content": prompt}])
    logger.info(
        "summarize_paper llm done: %s total=%.2fs",
        paper_id,
        time.monotonic() - started,
    )
    summary = json.loads(response)

    if pool:
        await db_papers.save_summary(pool, paper_id, summary)

    logger.info(
        "summarize_paper complete: %s total=%.2fs",
        paper_id,
        time.monotonic() - started,
    )
    return {
        "id": paper_id,
        "title": row["title"] if row else detail.get("title", ""),
        "authors": row["authors"] if row else detail.get("authors", []),
        "published": row["published"] if row else detail.get("published", ""),
        "url": row["url"] if row else detail.get("url", ""),
        "summary": summary.get("summary", ""),
        "keywords": summary.get("keywords", []),
    }


async def get_paper_content(
    arxiv_id: str | None = None,
    pdf_url: str | None = None,
) -> dict:
    started = time.monotonic()
    logger.info(
        "get_paper_content start: arxiv_id=%s pdf_url=%s",
        arxiv_id,
        pdf_url,
    )
    if arxiv_id:
        result = await summarize_paper(arxiv_id)
        logger.info(
            "get_paper_content done via arxiv_id=%s total=%.2fs",
            arxiv_id,
            time.monotonic() - started,
        )
        return result
    if pdf_url:
        host = (urlparse(pdf_url).hostname or "").lower()
        if host == "arxiv.org" or host.endswith(".arxiv.org"):
            match = _ARXIV_ID_RE.search(pdf_url)
            if match:
                result = await summarize_paper(match.group(1).removesuffix(".pdf"))
                logger.info(
                    "get_paper_content done via arxiv pdf_url=%s total=%.2fs",
                    pdf_url,
                    time.monotonic() - started,
                )
                return result
        result = await fetch_url(pdf_url)
        logger.info(
            "get_paper_content done via fetch_url=%s total=%.2fs",
            pdf_url,
            time.monotonic() - started,
        )
        return result
    logger.info("get_paper_content missing params total=%.2fs", time.monotonic() - started)
    return {"error": "Either arxiv_id or pdf_url must be provided"}
