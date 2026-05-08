# tools/web/server.py
import asyncio
import inspect
import json
from contextlib import asynccontextmanager

import mcp.types as types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

from .web_tools import fetch_url, web_search

server = Server("web")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="web_search",
            description="Search the web using Tavily. Use for blog posts, tutorials, news, benchmarks, industry announcements, or any topic not well covered by arxiv or GitHub.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "number of results to return, default 5",
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "'basic' (fast, default) or 'advanced' (deeper, slower)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="fetch_url",
            description="Fetch a web page and return its content as markdown. Use to read a specific URL found via web_search. Never use for arxiv.org URLs — use summarize_paper(paper_id) instead.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Non-arXiv URL to fetch"},
                },
                "required": ["url"],
            },
        ),
    ]


_TOOLS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    fn = _TOOLS.get(name)
    if fn is None:
        result = f"Unknown tool: {name}"
    else:
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**arguments)
            else:
                result = await asyncio.to_thread(fn, **arguments)  # type: ignore
        except Exception as e:
            result = {"error": str(e)}  # type: ignore
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


def build_http_app() -> Starlette:
    session_manager = StreamableHTTPSessionManager(app=server)

    @asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            yield

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    return Starlette(
        routes=[Mount("/mcp", app=handle_mcp)],
        lifespan=lifespan,
    )
