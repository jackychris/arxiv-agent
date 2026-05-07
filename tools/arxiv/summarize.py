# tools/arxiv/summarize.py
import json
import llm
from prompts import SUMMARIZE_PROMPT
from .paper_detail import get_paper_detail


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
