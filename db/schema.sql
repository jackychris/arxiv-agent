CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         JSONB NOT NULL DEFAULT '[]',
    published       TEXT,
    updated         TEXT,
    categories      JSONB NOT NULL DEFAULT '[]',
    url             TEXT,
    pdf_url         TEXT,
    venue           TEXT,
    citation_count  INTEGER,
    source_quality  TEXT,
    abstract        TEXT,

    -- LLM summary
    summary         TEXT,
    keywords        JSONB NOT NULL DEFAULT '[]',

    -- status flags
    has_content     BOOLEAN NOT NULL DEFAULT FALSE,
    has_summary     BOOLEAN NOT NULL DEFAULT FALSE,
    fetched_at      TIMESTAMPTZ,
    summarized_at   TIMESTAMPTZ,

    -- reserved for future semantic search
    embedding       vector(1536)
);

CREATE INDEX IF NOT EXISTS papers_fts_idx ON papers
    USING GIN (to_tsvector('english',
        coalesce(title, '') || ' ' || coalesce(abstract, '')));

CREATE INDEX IF NOT EXISTS papers_keywords_idx ON papers USING GIN (keywords);

CREATE TABLE IF NOT EXISTS memory (
    tool         TEXT NOT NULL,
    lesson       TEXT NOT NULL,
    added_at     TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tool, lesson)
);
