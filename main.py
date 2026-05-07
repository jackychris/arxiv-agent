# main.py
import asyncio
import logging
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from orchestrator import OrchestratorAgent
from researcher.server import build_app as build_research_app
from api.routes import router
from config import (
    PORT_MAIN,
    PORT_MCP,
    PORT_GITHUB_MCP,
    PORT_WEB_MCP,
    PORT_RESEARCH_AGENT,
    SERVER_START_TIMEOUT,
    validate_required,
)
from tools.arxiv.server import build_http_app as build_mcp_app
from tools.github.server import build_http_app as build_github_mcp_app
from tools.web.server import build_http_app as build_web_mcp_app


async def _serve(app, host: str, port: int, timeout: float = SERVER_START_TIMEOUT) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return server
        except OSError:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"Server on {host}:{port} failed to start within {timeout}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_required()
    mcp_server = await _serve(build_mcp_app(), "127.0.0.1", PORT_MCP)
    github_mcp_server = await _serve(build_github_mcp_app(), "127.0.0.1", PORT_GITHUB_MCP)
    web_mcp_server = await _serve(build_web_mcp_app(), "127.0.0.1", PORT_WEB_MCP)
    research_server = await _serve(build_research_app(), "127.0.0.1", PORT_RESEARCH_AGENT)
    app.state.agent = OrchestratorAgent()
    yield

    mcp_server.should_exit = True
    github_mcp_server.should_exit = True
    web_mcp_server.should_exit = True
    research_server.should_exit = True


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
