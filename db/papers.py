# db/papers.py
from typing import Any

import asyncpg


def _row(record: asyncpg.Record | None) -> dict | None:
    return dict(record) if record else None


async def get(pool: asyncpg.Pool, arxiv_id: str) -> dict | None:
    row = await pool.fetchrow("SELECT * FROM papers WHERE arxiv_id = $1", arxiv_id)
    return _row(row)


async def upsert_meta(pool: asyncpg.Pool, meta: dict[str, Any]) -> None:
    await pool.execute(
        """
        INSERT INTO papers (
            arxiv_id, title, authors, published, updated, categories,
            url, pdf_url, venue, citation_count, source_quality, abstract
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        ON CONFLICT (arxiv_id) DO UPDATE SET
            title          = EXCLUDED.title,
            authors        = EXCLUDED.authors,
            published      = COALESCE(EXCLUDED.published,      papers.published),
            updated        = COALESCE(EXCLUDED.updated,        papers.updated),
            categories     = EXCLUDED.categories,
            url            = COALESCE(EXCLUDED.url,            papers.url),
            pdf_url        = COALESCE(EXCLUDED.pdf_url,        papers.pdf_url),
            venue          = COALESCE(EXCLUDED.venue,          papers.venue),
            citation_count = COALESCE(EXCLUDED.citation_count, papers.citation_count),
            source_quality = COALESCE(EXCLUDED.source_quality, papers.source_quality),
            abstract       = COALESCE(EXCLUDED.abstract,       papers.abstract)
        """,
        meta.get("arxiv_id") or meta.get("id"),
        meta.get("title", ""),
        meta.get("authors", []),
        meta.get("published"),
        meta.get("updated"),
        meta.get("categories", []),
        meta.get("url"),
        meta.get("pdf_url"),
        meta.get("venue"),
        meta.get("citation_count"),
        meta.get("source_quality"),
        meta.get("abstract"),
    )


async def mark_content(pool: asyncpg.Pool, arxiv_id: str) -> None:
    await pool.execute(
        "UPDATE papers SET has_content = TRUE, fetched_at = NOW() WHERE arxiv_id = $1",
        arxiv_id,
    )


async def save_summary(pool: asyncpg.Pool, arxiv_id: str, summary: dict) -> None:
    await pool.execute(
        """
        UPDATE papers SET
            summary       = $2,
            keywords      = $3,
            has_summary   = TRUE,
            summarized_at = NOW()
        WHERE arxiv_id = $1
        """,
        arxiv_id,
        summary.get("summary"),
        summary.get("keywords", []),
    )


async def search_fts(pool: asyncpg.Pool, query: str, limit: int = 10) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT arxiv_id, title, authors, published, venue, citation_count,
               summary, keywords,
               ts_rank(
                   to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')),
                   plainto_tsquery('english', $1)
               ) AS rank
        FROM papers
        WHERE has_summary = TRUE
          AND to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,''))
              @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $2
        """,
        query,
        limit,
    )
    return [dict(r) for r in rows]
