from __future__ import annotations

from typing import Any

import asyncpg


def _row(record: asyncpg.Record | None) -> dict[str, Any] | None:
    return dict(record) if record else None


async def create(
    pool: asyncpg.Pool,
    *,
    run_id: str,
    guest_id: str,
    query: str,
    status: str = "running",
) -> None:
    await pool.execute(
        """
        INSERT INTO runs (run_id, guest_id, query, status)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (run_id) DO UPDATE
        SET guest_id = EXCLUDED.guest_id,
            query = COALESCE(NULLIF(runs.query, ''), EXCLUDED.query),
            status = EXCLUDED.status,
            updated_at = NOW()
        """,
        run_id,
        guest_id,
        query,
        status,
    )


async def update(
    pool: asyncpg.Pool,
    *,
    run_id: str,
    guest_id: str | None = None,
    rewritten_query: str | None = None,
    status: str | None = None,
    current_answer: str | None = None,
    error: dict[str, Any] | None = None,
    last_event_type: str | None = None,
    last_node: str | None = None,
    round_num: int | None = None,
    completed: bool = False,
) -> None:
    fields: list[str] = ["updated_at = NOW()"]
    values: list[Any] = [run_id]
    index = 2

    def push(expr: str, value: Any) -> None:
        nonlocal index
        fields.append(f"{expr} = ${index}")
        values.append(value)
        index += 1

    if guest_id is not None:
        push("guest_id", guest_id)
    if rewritten_query is not None:
        push("rewritten_query", rewritten_query)
    if status is not None:
        push("status", status)
    if current_answer is not None:
        push("current_answer", current_answer)
    if error is not None:
        push("error", error)
    if last_event_type is not None:
        push("last_event_type", last_event_type)
    if last_node is not None:
        push("last_node", last_node)
    if round_num is not None:
        push("round", round_num)
    if completed:
        fields.append("completed_at = NOW()")

    await pool.execute(
        f"""
        UPDATE runs
        SET {", ".join(fields)}
        WHERE run_id = $1
        """,
        *values,
    )


async def get(pool: asyncpg.Pool, *, run_id: str, guest_id: str | None = None) -> dict[str, Any] | None:
    if guest_id is None:
        row = await pool.fetchrow("SELECT * FROM runs WHERE run_id = $1", run_id)
    else:
        row = await pool.fetchrow(
            "SELECT * FROM runs WHERE run_id = $1 AND guest_id = $2",
            run_id,
            guest_id,
        )
    return _row(row)


async def list_by_guest(pool: asyncpg.Pool, *, guest_id: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT *
        FROM runs
        WHERE guest_id = $1
        ORDER BY updated_at DESC
        LIMIT $2
        """,
        guest_id,
        limit,
    )
    return [_row(row) for row in rows if row is not None]


async def delete(pool: asyncpg.Pool, *, run_id: str, guest_id: str) -> str | None:
    row = await pool.fetchrow(
        """
        DELETE FROM runs
        WHERE run_id = $1 AND guest_id = $2
        RETURNING run_id
        """,
        run_id,
        guest_id,
    )
    if row is None:
        return None
    return str(row["run_id"])
