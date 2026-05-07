# tools/github/_rate_limit.py
# GitHub Search API: 10 req/min unauthenticated, 30 req/min authenticated.
from config import GITHUB_TOKEN
from utils import AsyncRateLimiter

# Conservative delays: 7s without token (~8/min), 2.5s with token (~24/min)
_DELAY = 2.5 if GITHUB_TOKEN else 7.0
_limiter = AsyncRateLimiter(_DELAY)


async def github_api_call(coro):
    return await _limiter.call_awaitable(coro)
