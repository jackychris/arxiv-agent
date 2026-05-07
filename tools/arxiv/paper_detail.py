# tools/arxiv/paper_detail.py
import asyncio
import arxiv
from markitdown import MarkItDown
from config import ARXIV_FETCH_TIMEOUT

from ._rate_limit import arxiv_api_call

_md = MarkItDown()


async def _fetch_url(url: str) -> str | None:
    def _do():
        result = _md.convert_url(url)
        if result.text_content and len(result.text_content) > 500:
            return result.text_content
        return None

    try:
        return await asyncio.wait_for(asyncio.to_thread(_do), timeout=ARXIV_FETCH_TIMEOUT)
    except Exception:
        return None


async def get_paper_detail(paper_id: str) -> dict:
    results = await arxiv_api_call(
        lambda: list(arxiv.Client(page_size=1, num_retries=2, delay_seconds=3.0).results(
            arxiv.Search(id_list=[paper_id])
        ))
    )
    if not results:
        return {"error": f"Paper '{paper_id}' not found"}
    r = results[0]

    content = (
        await _fetch_url(f"https://arxiv.org/html/{paper_id}")
        or await _fetch_url(r.pdf_url)
        or r.summary
    )

    return {
        "id": r.entry_id.split("/")[-1],
        "title": r.title,
        "authors": [a.name for a in r.authors],
        "published": str(r.published.date()),
        "updated": str(r.updated.date()),
        "categories": r.categories,
        "url": r.entry_id,
        "pdf_url": r.pdf_url,
        "content": content,
    }
