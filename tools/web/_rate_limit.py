# tools/web/_rate_limit.py
# Tavily: monthly quota, no per-second hard limit.
# Small delay to be polite and avoid burst exhaustion.
from utils import AsyncRateLimiter

_DELAY = 1.0
_limiter = AsyncRateLimiter(_DELAY)


async def tavily_api_call(coro):
    return await _limiter.call_awaitable(coro)
