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

CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    guest_id           TEXT NOT NULL,
    query              TEXT NOT NULL,
    rewritten_query    TEXT NOT NULL DEFAULT '',
    status             TEXT NOT NULL DEFAULT 'running',
    current_answer     TEXT NOT NULL DEFAULT '',
    error              JSONB,
    last_event_type    TEXT,
    last_node          TEXT,
    round              INTEGER NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS runs_guest_created_idx
    ON runs (guest_id, created_at DESC);

CREATE INDEX IF NOT EXISTS runs_guest_updated_idx
    ON runs (guest_id, updated_at DESC);
