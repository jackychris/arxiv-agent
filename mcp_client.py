# mcp_client.py
import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from urllib.parse import urlparse

import httpx
import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT

from config import (
    ACADEMIC_MCP_URL,
    GITHUB_MCP_URL,
    MCP_ARXIV_MIN_INTERVAL_SECONDS,
    MCP_TOOL_TIMEOUT,
    WEB_MCP_URL,
)
from runtime import tool_cache
from runtime.policy import backoff_seconds, should_retry
from schemas import ErrorEnvelope, ToolResultEnvelope
from utils import AsyncRateLimiter

logger = logging.getLogger(__name__)

_ARXIV_MCP_TOOLS = {"search_arxiv", "download_arxiv", "read_arxiv_paper"}
_DEFAULT_TAVILY_EXTRACT_RESTRICTED_DOMAINS = {
    "blogger.com",
    "blogspot.com",
    "facebook.com",
    "fbcdn.net",
    "google.com",
    "google.com.hk",
    "google.com.tw",
    "googleapis.com",
    "googleusercontent.com",
    "googlevideo.com",
    "gstatic.com",
    "instagram.com",
    "medium.com",
    "reddit.com",
    "t.co",
    "telegram.org",
    "threads.net",
    "t.me",
    "twitter.com",
    "whatsapp.com",
    "wikimedia.org",
    "wikipedia.org",
    "x.com",
    "youtu.be",
    "youtube.com",
}
_arxiv_mcp_limiter = (
    AsyncRateLimiter(MCP_ARXIV_MIN_INTERVAL_SECONDS)
    if MCP_ARXIV_MIN_INTERVAL_SECONDS > 0
    else None
)


def load_servers() -> dict:
    with open("mcp_config.json") as f:
        servers = json.load(f)["servers"]
    env_overrides = {
        "academic": ACADEMIC_MCP_URL,
        "github": GITHUB_MCP_URL,
        "web": WEB_MCP_URL,
    }
    for name, url in env_overrides.items():
        if name in servers and url:
            servers[name]["url"] = url
    return servers


def tool_result(
    tool: str,
    *,
    ok: bool,
    data=None,
    error: dict | None = None,
    meta: dict | None = None,
) -> dict:
    return ToolResultEnvelope(ok=ok, tool=tool, data=data, error=error, meta=meta or {}).to_dict()  # type: ignore[arg-type]


def tool_error(
    tool: str,
    code: str,
    message: str,
    *,
    recoverable: bool = True,
    details: dict | None = None,
) -> dict:
    error = ErrorEnvelope(
        code=code,
        message=message,
        source=tool,
        recoverable=recoverable,
        details=details or {},
    )
    return tool_result(tool, ok=False, error=error.to_dict())


def wrap_tool_output(tool: str, output: str, *, is_error: bool = False) -> dict:
    parsed = _parse_tool_json(output)
    if is_error:
        return tool_error(tool, "MCP_TOOL_ERROR", _stringify(parsed), recoverable=True)
    if isinstance(parsed, dict) and "error" in parsed:
        return tool_error(
            tool,
            _infer_error_code(tool, str(parsed["error"])),
            str(parsed["error"]),
            recoverable=True,
            details={k: v for k, v in parsed.items() if k != "error"},
        )
    return tool_result(tool, ok=True, data=_normalize_tool_data(parsed))


def _normalize_tool_data(data):
    if isinstance(data, dict) and "result" in data and "results" not in data:
        value = data.get("result")
        if isinstance(value, list):
            return {**data, "results": value}
    return data


def encode_tool_result(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


def _parse_tool_json(output: str):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _infer_error_code(tool: str, message: str) -> str:
    lowered = message.lower()
    if "arxiv_rate_limit" in lowered or "arxiv api returned 429" in lowered:
        return "ARXIV_RATE_LIMIT"
    if "rate limit" in lowered or "too many requests" in lowered or " 429" in lowered:
        return "RATE_LIMIT"
    if "does not support arxiv" in lowered:
        return "ARXIV_URL_NOT_ALLOWED"
    if "timed out" in lowered or "timeout" in lowered:
        return "TOOL_TIMEOUT"
    if "not found" in lowered:
        return "TOOL_NOT_FOUND"
    return "TOOL_ERROR"


def _clean_tool_kwargs(kwargs: dict) -> dict:
    cleaned = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned


def _has_cacheable_data(data) -> bool:
    if data is None:
        return False
    if isinstance(data, str):
        return bool(data.strip())
    if isinstance(data, list):
        return bool(data)
    if isinstance(data, dict):
        for key in ("results", "result", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return bool(value)
        return bool(data)
    return True


def _normalize_domain(value: str) -> str:
    domain = value.strip().lower().rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _host_from_url(url: str) -> str:
    text = url.strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.netloc and "://" not in text:
        parsed = urlparse(f"//{text}")
    return _normalize_domain(parsed.hostname or "")


def _domain_matches(host: str, restricted_domain: str) -> bool:
    domain = _normalize_domain(restricted_domain)
    return host == domain or host.endswith(f".{domain}")


def _extract_tavily_urls(kwargs: dict) -> list[str]:
    urls: list[str] = []
    for key in ("url", "urls"):
        value = kwargs.get(key)
        if isinstance(value, str):
            urls.append(value)
        elif isinstance(value, list):
            urls.extend(item for item in value if isinstance(item, str))
    return urls


def _restricted_extract_target(
    kwargs: dict,
    restricted_domains: set[str] | list[str] | tuple[str, ...],
) -> tuple[str, str] | None:
    restricted = {
        _normalize_domain(str(domain))
        for domain in restricted_domains
        if str(domain).strip()
    }
    for url in _extract_tavily_urls(kwargs):
        host = _host_from_url(url)
        if not host:
            continue
        for domain in restricted:
            if _domain_matches(host, domain):
                return url, domain
    return None


_blocked_extract_target = _restricted_extract_target


def _is_arxiv_extract_url(url: str) -> bool:
    host = _host_from_url(url)
    path = urlparse(url.strip()).path.lower()
    if host == "arxiv.org":
        return path.startswith("/abs/") or path.startswith("/pdf/")
    return host == "export.arxiv.org"


def _tavily_extract_targets_arxiv(kwargs: dict) -> bool:
    return any(_is_arxiv_extract_url(url) for url in _extract_tavily_urls(kwargs))


class MCPClient:
    def __init__(self, allowed_tools: set[str] | None = None):
        self._stack = AsyncExitStack()
        self._tool_to_session: dict[str, ClientSession] = {}
        self._tool_schemas: list[types.Tool] = []
        self._tool_policies: dict[str, dict] = {}
        self._allowed = allowed_tools

    async def __aenter__(self):
        try:
            for server_name, cfg in load_servers().items():
                if "url" in cfg:
                    timeout = httpx.Timeout(
                        MCP_TOOL_TIMEOUT + 5.0,
                        read=max(MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_TOOL_TIMEOUT + 5.0),
                    )
                    http_client = await self._stack.enter_async_context(
                        httpx.AsyncClient(
                            trust_env=False,
                            follow_redirects=True,
                            timeout=timeout,
                        )
                    )
                    read, write, _ = await self._stack.enter_async_context(
                        streamable_http_client(cfg["url"], http_client=http_client)
                    )
                else:
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        cwd=cfg.get("cwd"),
                        env={**os.environ, **cfg.get("env", {})},
                    )
                    read, write = await self._stack.enter_async_context(stdio_client(params))

                session = await self._stack.enter_async_context(ClientSession(read, write))
                try:
                    await asyncio.wait_for(session.initialize(), timeout=MCP_TOOL_TIMEOUT)
                    result = await asyncio.wait_for(session.list_tools(), timeout=MCP_TOOL_TIMEOUT)
                except Exception as e:
                    raise RuntimeError(
                        f"MCP server {server_name!r} failed during initialize/list_tools: "
                        f"{e or type(e).__name__}"
                    ) from e

                actual_names = [tool.name for tool in result.tools]
                allowed_tools = set(cfg.get("allowed_tools") or [])
                if allowed_tools:
                    missing = sorted(allowed_tools - set(actual_names))
                    if missing:
                        raise RuntimeError(
                            f"MCP server {server_name!r} is missing required tools: {missing}. "
                            f"Actual tools: {actual_names}"
                        )
                for tool in result.tools:
                    if allowed_tools and tool.name not in allowed_tools:
                        continue
                    if tool.name in self._tool_to_session:
                        raise RuntimeError(
                            f"Tool name conflict: {tool.name!r} is registered by multiple servers"
                        )
                    self._tool_to_session[tool.name] = session
                    self._tool_schemas.append(tool)
                    self._tool_policies[tool.name] = {
                        "restricted_extract_domains": cfg.get(
                            "restricted_extract_domains",
                            cfg.get("blocked_extract_domains", []),
                        )
                    }

            if self._allowed is not None:
                self._tool_schemas = [t for t in self._tool_schemas if t.name in self._allowed]
                self._tool_to_session = {
                    k: v for k, v in self._tool_to_session.items() if k in self._allowed
                }
                self._tool_policies = {
                    k: v for k, v in self._tool_policies.items() if k in self._allowed
                }
            return self
        except Exception:
            await self._stack.aclose()
            raise

    async def __aexit__(self, *_):
        await self._stack.aclose()

    def get_tools_description(self) -> str:
        lines = []
        for tool in self._tool_schemas:
            lines.append(f"- {tool.name}: {tool.description}")
            schema = tool.inputSchema or {}
            props = schema.get("properties", {})
            required = schema.get("required", [])
            for param, info in props.items():
                req = "required" if param in required else "optional"
                lines.append(
                    f"  - {param} ({info.get('type', 'any')}, {req}): {info.get('description', '')}"
                )
        return "\n".join(lines)

    async def execute_tool(self, name: str, **kwargs) -> str:
        if name not in self._tool_to_session:
            return encode_tool_result(
                tool_error(
                    name,
                    "UNKNOWN_TOOL",
                    f"Unknown tool: {name!r}. Available: {list(self._tool_to_session.keys())}",
                    recoverable=False,
                )
            )

        kwargs = _clean_tool_kwargs(kwargs)

        cached = tool_cache.get(name, kwargs)
        if cached is not None:
            return cached

        session = self._tool_to_session[name]
        attempt = 1
        while True:
            envelope = await self._call_tool_once(name, session, kwargs)
            result_str = encode_tool_result(envelope)
            if envelope.get("ok"):
                if _has_cacheable_data(envelope.get("data")):
                    tool_cache.put(name, kwargs, result_str)
                return result_str
            if not should_retry(envelope, attempt):
                return result_str

            delay = backoff_seconds(attempt)
            logger.info(
                "Retrying MCP tool %s after attempt %d failed with %s; sleeping %.1fs",
                name,
                attempt,
                (envelope.get("error") or {}).get("code"),
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1

    async def _call_tool_once(self, name: str, session: ClientSession, kwargs: dict) -> dict:
        if name == "tavily_extract":
            restricted_domains = set(_DEFAULT_TAVILY_EXTRACT_RESTRICTED_DOMAINS)
            restricted_domains.update(
                self._tool_policies.get(name, {}).get("restricted_extract_domains") or []
            )
            restricted = _restricted_extract_target(kwargs, restricted_domains)
            if restricted:
                url, domain = restricted
                return tool_error(
                    name,
                    "RESTRICTED_EXTRACT_DOMAIN",
                    (
                        f"Refusing to extract {url!r}: domain matches restricted domain "
                        f"{domain!r}. Use another source or a stable mirror/page."
                    ),
                    recoverable=False,
                    details={"url": url, "restricted_domain": domain},
                )

        try:
            call = session.call_tool(name, kwargs)
            use_arxiv_limiter = name in _ARXIV_MCP_TOOLS or (
                name == "tavily_extract" and _tavily_extract_targets_arxiv(kwargs)
            )
            if use_arxiv_limiter and _arxiv_mcp_limiter is not None:
                call = _arxiv_mcp_limiter.call_awaitable(call)
            result = await asyncio.wait_for(
                call, timeout=MCP_TOOL_TIMEOUT
            )
        except Exception as e:
            return tool_error(
                name,
                "MCP_TOOL_TIMEOUT",
                str(e) or type(e).__name__,
                recoverable=True,
                details={"exception_type": type(e).__name__},
            )

        return self._wrap_call_tool_result(name, result)

    def _wrap_call_tool_result(self, name: str, result) -> dict:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return tool_result(
                name,
                ok=not getattr(result, "isError", False),
                data=_normalize_tool_data(structured),
            )

        texts = []
        for block in result.content:
            if getattr(block, "type", None) != "text":
                return tool_error(
                    name,
                    "MCP_INVALID_CONTENT",
                    f"Tool {name!r} returned non-text content: {getattr(block, 'type', None)}",
                    recoverable=False,
                )
            texts.append(block.text)
        output = "\n".join(texts)
        return wrap_tool_output(name, output, is_error=getattr(result, "isError", False))
