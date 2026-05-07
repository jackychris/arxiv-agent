# tools/arxiv/arxiv_search.py
from datetime import UTC, datetime, timedelta

import arxiv

from ._rate_limit import arxiv_api_call


async def search_arxiv(
    query: str,
    max_results: int = 5,
    days: int | None = None,
    author: str | None = None,
) -> list[dict]:
    max_results = min(max_results, 10)
    if author is not None:
        query = f"au:{author} AND {query}" if query else f"au:{author}"
    if days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        date_str = cutoff.strftime("%Y%m%d%H%M%S")
        query = f"{query} submittedDate:[{date_str} TO 99991231235959]"
        sort_by = arxiv.SortCriterion.SubmittedDate
        sort_order = arxiv.SortOrder.Descending
    else:
        sort_by = arxiv.SortCriterion.Relevance
        sort_order = arxiv.SortOrder.Descending

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    def do_fetch():
        client = arxiv.Client(page_size=max_results, num_retries=2, delay_seconds=3.0)
        return [
            {
                "id": r.entry_id.split("/")[-1],
                "title": r.title,
                "summary": r.summary,
                "url": r.entry_id,
                "authors": [a.name for a in r.authors[:3]],
                "published": str(r.published.date()),
            }
            for r in client.results(search)
        ]

    return await arxiv_api_call(do_fetch)
