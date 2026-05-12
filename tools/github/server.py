# tools/github/server.py
import asyncio
import inspect
import json
import sys
from contextlib import asynccontextmanager

import mcp.types as types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

from .github_tools import get_repo_readme, search_code, search_repos

server = Server("github")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_repos",
            description="Search GitHub repositories by keyword. Returns stars, forks, language, topics, last update. Use to find code implementations of papers, official repos, or libraries for a topic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "search keywords, e.g. paper title or topic",
                    },
                    "language": {
                        "type": "string",
                        "description": "filter by programming language, e.g. 'python'",
                    },
                    "sort": {
                        "type": "string",
                        "description": "sort by 'stars' (default) or 'updated'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "number of repos to return, default 5",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_repo_readme",
            description="Fetch the README of a GitHub repository. Use after search_repos to understand what a repo does, its usage, and what paper or algorithm it implements.",
            inputSchema={
                "type": "object",
                "properties": {
                    "full_name": {
                        "type": "string",
                        "description": "repository full name, e.g. 'huggingface/peft'",
                    },
                },
                "required": ["full_name"],
            },
        ),
        types.Tool(
            name="search_code",
            description="Search for code snippets on GitHub. Use to find specific algorithm implementations, class definitions, or usage examples when you know what to look for in code.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "code search query, e.g. 'class LoRALayer'",
                    },
                    "language": {"type": "string", "description": "filter by programming language"},
                    "max_results": {
                        "type": "integer",
                        "description": "number of results to return, default 5",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


_TOOLS = {
    "search_repos": search_repos,
    "get_repo_readme": get_repo_readme,
    "search_code": search_code,
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


def run_http(port: int = 8767):
    import uvicorn

    uvicorn.run(build_http_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    if "--http" in sys.argv:
        run_http()
