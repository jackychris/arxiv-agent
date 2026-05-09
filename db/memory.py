# db/memory.py
import asyncpg

from config import MEMORY_MAX_ACTIVE, MEMORY_MAX_STORED


async def add(pool: asyncpg.Pool, tool: str, lesson: str) -> None:
    await pool.execute(
        """
        INSERT INTO memory (tool, lesson)
        VALUES ($1, $2)
        ON CONFLICT (tool, lesson) DO UPDATE SET last_used_at = NOW()
        """,
        tool,
        lesson,
    )
    # Evict LRU entries beyond the stored limit
    await pool.execute(
        """
        DELETE FROM memory
        WHERE tool = $1
          AND lesson NOT IN (
              SELECT lesson FROM memory
              WHERE tool = $1
              ORDER BY last_used_at DESC
              LIMIT $2
          )
        """,
        tool,
        MEMORY_MAX_STORED,
    )


async def get(pool: asyncpg.Pool, tool: str) -> list[str]:
    rows = await pool.fetch(
        """
        UPDATE memory SET last_used_at = NOW()
        WHERE tool = $1
          AND lesson IN (
              SELECT lesson FROM memory
              WHERE tool = $1
              ORDER BY last_used_at DESC
              LIMIT $2
          )
        RETURNING lesson
        """,
        tool,
        MEMORY_MAX_ACTIVE,
    )
    return [r["lesson"] for r in rows]


async def get_all(pool: asyncpg.Pool, tools: list[str]) -> dict[str, list[str]]:
    return {tool: await get(pool, tool) for tool in tools}
