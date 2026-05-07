"""Process-level LRU cache for tool results.

Parallel subagents frequently query the same paper or search term.
Caching successful results eliminates redundant network calls and
avoids burning rate-limit quota on identical inputs.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 256

# tool:canonical_kwargs_json -> encoded result string
_cache: OrderedDict[str, str] = OrderedDict()


def _key(tool: str, kwargs: dict) -> str:
    return tool + ":" + json.dumps(kwargs, sort_keys=True, ensure_ascii=False)


def get(tool: str, kwargs: dict) -> str | None:
    k = _key(tool, kwargs)
    if k not in _cache:
        return None
    _cache.move_to_end(k)
    logger.debug("Cache hit: %s %s", tool, _short(kwargs))
    return _cache[k]


def put(tool: str, kwargs: dict, result: str) -> None:
    k = _key(tool, kwargs)
    _cache[k] = result
    _cache.move_to_end(k)
    if len(_cache) > _MAX_ENTRIES:
        evicted, _ = _cache.popitem(last=False)
        logger.debug("Cache evicted: %s", evicted[:60])


def clear() -> None:
    _cache.clear()


def size() -> int:
    return len(_cache)


def _short(kwargs: dict) -> str:
    s = json.dumps(kwargs, ensure_ascii=False)
    return s[:80] + "..." if len(s) > 80 else s
