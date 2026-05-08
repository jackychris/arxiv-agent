from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return default
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def retry_schedule(delays: list[float], *, initial_immediate: bool = True) -> list[float]:
    schedule = list(delays)
    if initial_immediate:
        schedule.insert(0, 0.0)
    return schedule


def rate_limit_wait(elapsed: float, min_interval: float) -> float:
    return max(0.0, min_interval - elapsed)


class SyncRateLimiter:
    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        with self._lock:
            if self._last_call is not None:
                wait = rate_limit_wait(self._clock() - self._last_call, self.min_interval)
                if wait > 0:
                    self._sleep(wait)
            result = fn(*args, **kwargs)
            self._last_call = self._clock()
            return result


class AsyncRateLimiter:
    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last_call: float | None = None

    async def call_awaitable(self, awaitable: Awaitable[T]) -> T:
        async with self._lock:
            if self._last_call is not None:
                wait = rate_limit_wait(self._clock() - self._last_call, self.min_interval)
                if wait > 0:
                    await self._sleep(wait)
            result = await awaitable
            self._last_call = self._clock()
            return result


def retry_sync(
    fn: Callable[[], T],
    *,
    retry_delays: list[float],
    is_retryable_exception: Callable[[BaseException], bool],
    exhausted_exception: Callable[[BaseException | None, int], BaseException],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    schedule = retry_schedule(retry_delays)
    last_exc: BaseException | None = None
    for wait in schedule:
        if wait:
            sleep(wait)
        try:
            return fn()
        except BaseException as exc:
            if not is_retryable_exception(exc):
                raise
            last_exc = exc
    raise exhausted_exception(last_exc, len(schedule))


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    retry_delays: list[float],
    is_retryable_exception: Callable[[BaseException], bool],
    exhausted_exception: Callable[[BaseException | None, int], BaseException],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    schedule = retry_schedule(retry_delays)
    last_exc: BaseException | None = None
    for wait in schedule:
        if wait:
            await sleep(wait)
        try:
            return await fn()
        except BaseException as exc:
            if not is_retryable_exception(exc):
                raise
            last_exc = exc
    raise exhausted_exception(last_exc, len(schedule))
