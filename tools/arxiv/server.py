# tools/arxiv/server.py
import asyncio
import inspect
import json
import sys
from contextlib import asynccontextmanager

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

from .arxiv_search import search_arxiv
from .semantic_scholar import search_semantic_scholar
from .summarize import summarize_paper

server = Server("arxiv")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_arxiv",
            description="Search arxiv for academic papers. Use only when you need papers from the last few weeks (real-time index). For general searches prefer search_semantic_scholar which is faster. Supports filtering by author and date range.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search keywords"},
                    "max_results": {
                        "type": "integer",
                        "description": "number of papers to return, default 5",
                    },
                    "days": {
                        "type": "integer",
                        "description": "if set, only return papers from the last N days",
                    },
                    "author": {"type": "string", "description": "if set, filter by author name"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="summarize_paper",
            description="Fetch full paper content by arxiv ID and return a structured summary: background, method, results, limitations, keywords, datasets. Use when you need deeper detail beyond what search results provide.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "arxiv paper ID, e.g. '2005.11401'",
                    },
                },
                "required": ["paper_id"],
            },
        ),
        types.Tool(
            name="search_semantic_scholar",
            description="Search academic papers via Semantic Scholar. Faster than search_arxiv, includes citation counts and AI-generated TLDRs. Use this for most searches. Use search_arxiv only when you need papers from the last few weeks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search keywords"},
                    "max_results": {
                        "type": "integer",
                        "description": "number of papers to return, default 5",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "if set, only return papers from this year onwards, e.g. 2023",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


_TOOLS = {
    "search_arxiv": search_arxiv,
    "summarize_paper": summarize_paper,
    "search_semantic_scholar": search_semantic_scholar,
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


async def run_stdio():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


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


def run_http(port: int = 8765):
    import uvicorn

    uvicorn.run(build_http_app(), host="127.0.0.1", port=port)


if __name__ == "__main__":
    if "--http" in sys.argv:
        run_http()
    else:
        asyncio.run(run_stdio())
