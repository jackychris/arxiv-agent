# db/pool.py
import json
from pathlib import Path

import asyncpg

from config import DATABASE_URL, DISABLE_DB

_pool: asyncpg.Pool | None = None


async def _setup_conn(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    await conn.set_type_codec("json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def init() -> None:
    global _pool
    if DISABLE_DB:
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10, init=_setup_conn)
    schema = (Path(__file__).parent / "schema.sql").read_text()
    async with _pool.acquire() as conn:
        await conn.execute(schema)


async def get() -> asyncpg.Pool | None:
    return _pool


async def close() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
