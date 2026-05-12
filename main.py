# main.py
import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import httpx
from a2a.client.client_factory import ClientConfig, ClientFactory

import db
from api.routes import router
from config import (
    A2A_CLIENT_TIMEOUT,
    ARXIV_MCP_HOST,
    DATABASE_URL,
    DISABLE_DB,
    GITHUB_MCP_HOST,
    PORT_MAIN,
    PORT_MCP,
    PORT_GITHUB_MCP,
    PORT_RESEARCH_AGENT,
    PORT_WEB_MCP,
    RESEARCH_AGENT_HOST,
    SERVER_START_TIMEOUT,
    WEB_MCP_HOST,
    validate_required,
)
from graph import build_graph
DEPENDENCIES = (
    ("arxiv MCP", ARXIV_MCP_HOST, PORT_MCP, None),
    ("github MCP", GITHUB_MCP_HOST, PORT_GITHUB_MCP, None),
    ("web MCP", WEB_MCP_HOST, PORT_WEB_MCP, None),
    ("research agent", RESEARCH_AGENT_HOST, PORT_RESEARCH_AGENT, None),
)


async def _wait_for_http_dependency(
    name: str,
    host: str,
    port: int,
    ready_path: str | None,
    timeout: float = SERVER_START_TIMEOUT,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_error: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            if not ready_path:
                return
            async with httpx.AsyncClient(timeout=1.0, trust_env=False) as client:
                response = await client.get(f"http://{host}:{port}{ready_path}")
            if response.status_code < 500:
                return
            last_error = f"HTTP {response.status_code}"
        except (OSError, httpx.HTTPError) as exc:
            last_error = str(exc)
        await asyncio.sleep(0.2)
    raise TimeoutError(
        f"Dependency {name} at {host}:{port} was not ready within {timeout}s"
        + (f" (last error: {last_error})" if last_error else "")
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_required()
    await db.init()
    for name, host, port, ready_path in DEPENDENCIES:
        await _wait_for_http_dependency(name, host, port, ready_path)
    async with httpx.AsyncClient(timeout=A2A_CLIENT_TIMEOUT, trust_env=False) as http:
        if DISABLE_DB:
            app.state.checkpointer = InMemorySaver()
            app.state.graph = build_graph().compile(checkpointer=app.state.checkpointer)
            app.state.factory = ClientFactory(ClientConfig(httpx_client=http))
            yield
        else:
            async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
                await checkpointer.setup()
                app.state.checkpointer = checkpointer
                app.state.graph = build_graph().compile(checkpointer=checkpointer)
                app.state.factory = ClientFactory(ClientConfig(httpx_client=http))
                yield
    await db.close()


app = FastAPI(title="arxiv Research Agent", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=PORT_MAIN, reload=False)
