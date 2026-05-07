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
from .citations import get_citation_info
from .summarize import summarize_paper

server = Server("arxiv")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_arxiv",
            description="Search arxiv for academic papers. Supports filtering by author and recency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search keywords"},
                    "max_results": {"type": "integer", "description": "number of papers to return, default 5"},
                    "days": {"type": "integer", "description": "if set, only return papers from the last N days"},
                    "author": {"type": "string", "description": "if set, filter by author name"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="summarize_paper",
            description="Fetch a paper by ID and return a structured summary: background, method, results, limitations, keywords, datasets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arxiv paper ID, e.g. '2005.11401'"},
                },
                "required": ["paper_id"],
            },
        ),
        types.Tool(
            name="get_citation_info",
            description="Get citation count for a paper, and optionally the list of papers it references or papers that cite it. Uses Semantic Scholar.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arxiv paper ID, e.g. '2005.11401'"},
                    "include_references": {"type": "boolean", "description": "if true, return papers this paper cites, default false"},
                    "include_citations": {"type": "boolean", "description": "if true, return papers that cite this paper, default false"},
                    "max_results": {"type": "integer", "description": "max papers to return per list, default 10"},
                },
                "required": ["paper_id"],
            },
        ),
    ]


_TOOLS = {
    "search_arxiv": search_arxiv,
    "summarize_paper": summarize_paper,
    "get_citation_info": get_citation_info,
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
                result = await asyncio.to_thread(fn, **arguments)
        except Exception as e:
            result = {"error": str(e)}
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
