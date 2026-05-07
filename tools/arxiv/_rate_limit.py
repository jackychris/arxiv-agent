# tools/arxiv/_rate_limit.py
# Shared rate limit for all arxiv API calls (export.arxiv.org/api/query).
# arxiv policy: no more than 1 request per 3 seconds.
import asyncio
import urllib.error

from config import ARXIV_429_RETRY_DELAYS

_lock = asyncio.Lock()
_sleep = asyncio.sleep


class ArxivRateLimitError(RuntimeError):
    def __init__(self, attempts: int):
        super().__init__(f"ARXIV_RATE_LIMIT: arxiv API returned 429 after {attempts} attempts")
        self.attempts = attempts


async def arxiv_api_call(fn, *args, **kwargs):
    delays = [0.0] + list(ARXIV_429_RETRY_DELAYS)
    for wait in delays:
        if wait:
            await _sleep(wait)
        async with _lock:
            try:
                result = await asyncio.to_thread(fn, *args, **kwargs)
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                continue
            return result
    raise ArxivRateLimitError(attempts=len(delays))
