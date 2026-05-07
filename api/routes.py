# api/routes.py
import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.schemas import ChatRequest, ChatResponse
from config import SSE_HEARTBEAT_INTERVAL
from orchestrator.events import encode_sse, heartbeat

router = APIRouter()

_START_TIME = time.time()


@router.get("/healthz")
async def healthz(request: Request):
    agent_ready = hasattr(request.app.state, "agent")
    if not agent_ready:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok", "uptime_s": round(time.time() - _START_TIME)}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    answer = await request.app.state.agent.run(body.query)
    return ChatResponse(answer=answer)


@router.get("/stream")
async def stream(query: str, request: Request):
    queue: asyncio.Queue = asyncio.Queue()

    async def generate():
        agent_task = asyncio.create_task(request.app.state.agent.run_streaming(query, queue))
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_INTERVAL)
                except TimeoutError:
                    yield heartbeat()
                    continue
                if event is None:
                    break
                yield encode_sse(event)
        finally:
            agent_task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
