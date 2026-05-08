# mcp_client.py
import asyncio
import json
import os
from contextlib import AsyncExitStack

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from config import MCP_TOOL_TIMEOUT
from runtime import tool_cache
from schemas import ErrorEnvelope, ToolResultEnvelope


def load_servers() -> dict:
    with open("mcp_config.json") as f:
        return json.load(f)["servers"]


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
    return tool_result(tool, ok=True, data=parsed)


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
    if "fetch_url does not support arxiv" in lowered:
        return "ARXIV_URL_NOT_ALLOWED"
    if "timed out" in lowered or "timeout" in lowered:
        return "TOOL_TIMEOUT"
    if "not found" in lowered:
        return "TOOL_NOT_FOUND"
    return "TOOL_ERROR"


class MCPClient:
    def __init__(self):
        self._stack = AsyncExitStack()
        self._tool_to_session: dict[str, ClientSession] = {}
        self._tool_schemas: list[types.Tool] = []

    async def __aenter__(self):
        for _, cfg in load_servers().items():
            if "url" in cfg:
                read, write, _ = await self._stack.enter_async_context(
                    streamable_http_client(cfg["url"])
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
            await session.initialize()
            result = await session.list_tools()
            for tool in result.tools:
                if tool.name in self._tool_to_session:
                    raise RuntimeError(
                        f"Tool name conflict: {tool.name!r} is registered by multiple servers"
                    )
                self._tool_to_session[tool.name] = session
                self._tool_schemas.append(tool)
        return self

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

        cached = tool_cache.get(name, kwargs)
        if cached is not None:
            return cached

        session = self._tool_to_session[name]
        try:
            result = await asyncio.wait_for(
                session.call_tool(name, kwargs), timeout=MCP_TOOL_TIMEOUT
            )
        except Exception as e:
            return encode_tool_result(
                tool_error(
                    name,
                    "MCP_TOOL_TIMEOUT",
                    str(e) or type(e).__name__,
                    recoverable=True,
                    details={"exception_type": type(e).__name__},
                )
            )

        envelope = self._wrap_call_tool_result(name, result)
        result_str = encode_tool_result(envelope)
        if envelope.get("ok"):
            tool_cache.put(name, kwargs, result_str)
        return result_str

    def _wrap_call_tool_result(self, name: str, result) -> dict:
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
