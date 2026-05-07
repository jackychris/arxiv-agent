# tools/arxiv/citations.py
import httpx

SS_API = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "title,authors,year,externalIds,abstract"


def _format(paper: dict) -> dict:
    ext = paper.get("externalIds") or {}
    return {
        "title": paper.get("title", ""),
        "authors": [a["name"] for a in paper.get("authors", [])[:3]],
        "year": paper.get("year"),
        "arxiv_id": ext.get("ArXiv"),
        "abstract": (paper.get("abstract") or "")[:300],
    }


def get_citation_info(
    paper_id: str,
    include_references: bool = False,
    include_citations: bool = False,
    max_results: int = 10,
) -> dict:
    """Get citation count and optionally the reference/citation lists for a paper."""
    resp = httpx.get(
        f"{SS_API}/paper/arXiv:{paper_id}",
        params={"fields": "citationCount"},
    )
    resp.raise_for_status()
    result = {"citation_count": resp.json().get("citationCount")}

    if include_references:
        resp = httpx.get(
            f"{SS_API}/paper/arXiv:{paper_id}/references",
            params={"fields": PAPER_FIELDS, "limit": max_results},
        )
        resp.raise_for_status()
        result["references"] = [_format(item["citedPaper"]) for item in resp.json().get("data", [])]

    if include_citations:
        resp = httpx.get(
            f"{SS_API}/paper/arXiv:{paper_id}/citations",
            params={"fields": PAPER_FIELDS, "limit": max_results},
        )
        resp.raise_for_status()
        result["citations"] = [_format(item["citingPaper"]) for item in resp.json().get("data", [])]

    return result
