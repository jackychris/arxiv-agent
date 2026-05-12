# api/routes.py
import asyncio
import contextlib
import dataclasses
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.types import Command

from api.schemas import ChatRequest, ChatResponse
from config import SSE_DISCONNECT_GRACE, SSE_HEARTBEAT_INTERVAL, SSE_REVIEW_HEARTBEAT_INTERVAL
from db import pool as db_pool
from db import runs as db_runs
from graph import errors as err
from graph.events import encode_sse, error as evt_error, heartbeat, human_review as evt_human_review
from graph.state import ResearchState
from runtime import run_context

router = APIRouter()
logger = logging.getLogger(__name__)

_START_TIME = time.time()
_GUEST_COOKIE = "guest_id"
_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _unwrap_custom_event(chunk):
    if (
        isinstance(chunk, (list, tuple))
        and len(chunk) == 2
        and chunk[0] == "custom"
        and isinstance(chunk[1], dict)
    ):
        return chunk[1]
    return chunk


def _initial_state(query: str, run_id: str) -> ResearchState:
    return {
        "query": query,
        "rewritten_query": "",
        "context_id": run_id,
        "human_in_loop": True,
        "tasks": [],
        "subagent_results": {},
        "known_citations": [],
        "round_histories": [],
        "current_answer": "",
        "round": 0,
        "critique_ok": False,
        "critique_gaps": [],
        "planning_guidance": "",
        "plan_review_action": "",
        "plan_review_feedback": "",
        "critique_review_action": "",
        "critique_review_feedback": "",
    }


def _new_guest_id() -> str:
    return f"guest-{uuid.uuid4().hex}"


def _get_guest_id(request: Request) -> str | None:
    value = request.cookies.get(_GUEST_COOKIE)
    if not value:
        return None
    guest_id = str(value).strip()
    return guest_id or None


def _set_guest_cookie(response: JSONResponse, guest_id: str) -> None:
    response.set_cookie(
        _GUEST_COOKIE,
        guest_id,
        max_age=60 * 60 * 24 * 180,
        samesite="lax",
        httponly=False,
    )


def _get_abort_event(app, run_id: str) -> asyncio.Event:
    abort_events = getattr(app.state, "abort_events", None)
    if abort_events is None:
        abort_events = {}
        app.state.abort_events = abort_events
    event = abort_events.get(run_id)
    if event is None:
        event = asyncio.Event()
        abort_events[run_id] = event
    return event


def _clear_abort_event(app, run_id: str) -> None:
    abort_events = getattr(app.state, "abort_events", None)
    if abort_events is not None:
        abort_events.pop(run_id, None)


def _set_active_stream_token(app, run_id: str, token: str) -> None:
    tokens = getattr(app.state, "stream_tokens", None)
    if tokens is None:
        tokens = {}
        app.state.stream_tokens = tokens
    tokens[run_id] = token


def _is_active_stream_token(app, run_id: str, token: str) -> bool:
    tokens = getattr(app.state, "stream_tokens", None)
    if not isinstance(tokens, dict):
        return False
    return tokens.get(run_id) == token


def _clear_active_stream_token(app, run_id: str, token: str) -> None:
    tokens = getattr(app.state, "stream_tokens", None)
    if not isinstance(tokens, dict):
        return
    if tokens.get(run_id) == token:
        tokens.pop(run_id, None)


def _checkpoint_to_dict(run_id: str, checkpoint) -> dict:
    channel_values = checkpoint.checkpoint.get("channel_values", {})
    return {
        "run_id": run_id,
        "checkpoint_id": checkpoint.config["configurable"]["checkpoint_id"],
        "parent_checkpoint_id": (
            checkpoint.parent_config["configurable"]["checkpoint_id"]
            if checkpoint.parent_config
            else None
        ),
        "metadata": _to_jsonable(checkpoint.metadata),
        "pending_writes": len(checkpoint.pending_writes),
        "channel_summary": _summarize_channel_values(channel_values),
    }


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return {"type": type(value).__name__, "repr": repr(value)}


def _preview(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _summarize_task(task) -> dict:
    if not isinstance(task, Mapping):
        return {"type": type(task).__name__, "repr": repr(task)}
    return {
        "id": task.get("id"),
        "effort": task.get("effort"),
        "mission_preview": _preview(str(task.get("mission", "")), 160),
    }


def _summarize_subagent_result(item) -> dict:
    if not isinstance(item, Mapping):
        return {"type": type(item).__name__, "repr": repr(item)}
    error = item.get("error")
    error_code = error.get("code") if isinstance(error, Mapping) else None
    return {
        "task_id": item.get("task_id"),
        "ok": item.get("ok"),
        "error_code": error_code,
        "result_preview": _preview(str(item.get("result", "")), 160),
    }


def _summarize_citation(item) -> dict:
    if not isinstance(item, Mapping):
        return {"type": type(item).__name__, "repr": repr(item)}
    return {
        "ref_num": item.get("ref_num"),
        "key": item.get("key"),
        "title": _preview(str(item.get("title", "")), 120),
        "cited": item.get("cited"),
    }


def _summarize_channel_values(channel_values) -> dict:
    values = _to_jsonable(channel_values)
    if not isinstance(values, Mapping):
        return {"type": type(values).__name__, "repr": repr(values)}

    summary: dict[str, object] = {}
    if "__start__" in values:
        start = values.get("__start__")
        if isinstance(start, Mapping):
            summary["input_keys"] = sorted(str(k) for k in start.keys())
            summary["input_query"] = start.get("query")

    summary["query"] = values.get("query")
    summary["rewritten_query"] = values.get("rewritten_query")
    summary["context_id"] = values.get("context_id")
    summary["round"] = values.get("round")
    summary["human_in_loop"] = values.get("human_in_loop")
    summary["critique_ok"] = values.get("critique_ok")
    summary["plan_review_action"] = values.get("plan_review_action")
    summary["critique_review_action"] = values.get("critique_review_action")

    current_answer = values.get("current_answer")
    if isinstance(current_answer, str):
        summary["current_answer"] = {
            "length": len(current_answer),
            "preview": _preview(current_answer),
        }

    planning_guidance = values.get("planning_guidance")
    if isinstance(planning_guidance, str) and planning_guidance:
        summary["planning_guidance"] = _preview(planning_guidance)

    critique_gaps = values.get("critique_gaps")
    if isinstance(critique_gaps, Sequence) and not isinstance(critique_gaps, (str, bytes, bytearray)):
        summary["critique_gaps"] = {
            "count": len(critique_gaps),
            "items": [_preview(str(item), 160) for item in list(critique_gaps)[:5]],
        }

    tasks = values.get("tasks")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        task_list = list(tasks)
        summary["tasks"] = {
            "count": len(task_list),
            "items": [_summarize_task(task) for task in task_list[:5]],
        }

    subagent_results = values.get("subagent_results")
    if isinstance(subagent_results, Mapping):
        items = list(subagent_results.values())
        summary["subagent_results"] = {
            "count": len(items),
            "items": [_summarize_subagent_result(item) for item in items[:5]],
        }

    known_citations = values.get("known_citations")
    if isinstance(known_citations, Sequence) and not isinstance(known_citations, (str, bytes, bytearray)):
        citation_list = list(known_citations)
        summary["known_citations"] = {
            "count": len(citation_list),
            "items": [_summarize_citation(item) for item in citation_list[:8]],
        }

    round_histories = values.get("round_histories")
    if isinstance(round_histories, Sequence) and not isinstance(round_histories, (str, bytes, bytearray)):
        history_list = list(round_histories)
        summary["round_histories"] = {
            "count": len(history_list),
            "items": [_preview(str(item), 160) for item in history_list[:3]],
        }

    pregel_tasks = values.get("__pregel_tasks")
    if isinstance(pregel_tasks, Sequence) and not isinstance(pregel_tasks, (str, bytes, bytearray)):
        summary["pregel_tasks"] = {
            "count": len(list(pregel_tasks)),
        }

    branch_keys = sorted(str(key) for key in values.keys() if str(key).startswith("branch:to:"))
    if branch_keys:
        summary["branch_keys"] = branch_keys

    summary["available_keys"] = sorted(str(k) for k in values.keys())
    return summary


def _with_full_checkpoint(payload: dict, checkpoint, *, full: bool) -> dict:
    if full:
        payload["channel_values"] = _to_jsonable(checkpoint.checkpoint.get("channel_values", {}))
    return payload


def _run_to_dict(row: Mapping[str, object], *, include_answer: bool = False) -> dict[str, object]:
    answer = str(row.get("current_answer") or "")
    status = row.get("status")
    payload: dict[str, object] = {
        "run_id": row.get("run_id"),
        "guest_id": row.get("guest_id"),
        "query": row.get("query"),
        "rewritten_query": row.get("rewritten_query"),
        "status": status,
        "last_event_type": row.get("last_event_type"),
        "last_node": row.get("last_node"),
        "round": row.get("round"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "completed_at": row.get("completed_at"),
        "answer_preview": _preview(answer, 240) if answer else "",
        "answer_length": len(answer),
        "has_answer": bool(answer),
        "error": row.get("error"),
    }
    if include_answer:
        payload["current_answer"] = answer
    return payload


async def _persist_run_start(*, run_id: str, guest_id: str, query: str) -> None:
    pool = await db_pool.get()
    if pool is None:
        return
    await db_runs.create(pool, run_id=run_id, guest_id=guest_id, query=query, status="running")


async def _persist_run_resume(*, run_id: str, guest_id: str) -> None:
    pool = await db_pool.get()
    if pool is None:
        return
    await db_runs.update(
        pool,
        run_id=run_id,
        guest_id=guest_id,
        status="running",
        last_event_type="resume",
    )


async def _persist_run_disconnect(*, run_id: str) -> None:
    pool = await db_pool.get()
    if pool is None:
        return
    await db_runs.update(
        pool,
        run_id=run_id,
        status="disconnected",
        last_event_type="disconnect",
    )


async def _delete_run_artifacts(request: Request, run_id: str) -> None:
    abort_event = _get_abort_event(request.app, run_id)
    abort_event.set()
    run_context.clear(run_id)
    _clear_abort_event(request.app, run_id)

    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "adelete_thread"):
        with contextlib.suppress(Exception):
            await checkpointer.adelete_thread(run_id)

    trace_path = _RUNS_DIR / f"{run_id}.jsonl"
    with contextlib.suppress(FileNotFoundError):
        trace_path.unlink()


async def _persist_run_event(run_id: str, event: dict) -> None:
    pool = await db_pool.get()
    if pool is None:
        return

    event_type = event.get("type")
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    patch: dict[str, object] = {"last_event_type": str(event_type or "")}

    if isinstance(data.get("round"), int):
        patch["round_num"] = int(data["round"])

    if event_type == "node_result":
        node = str(data.get("node") or "")
        patch["last_node"] = node
        if node in {
            "init",
            "rewrite_query",
            "plan_tasks",
            "collect_results",
            "synthesize_answer",
            "critique_answer",
            "research_task",
        }:
            patch["status"] = "running"
        if node == "rewrite_query":
            rewritten = data.get("rewritten_query")
            if isinstance(rewritten, str):
                patch["rewritten_query"] = rewritten
        elif node == "synthesize_answer":
            content = event.get("content")
            if isinstance(content, str):
                patch["current_answer"] = content
        elif node == "finalize":
            content = event.get("content")
            if isinstance(content, str):
                patch["current_answer"] = content

    elif event_type == "human_review":
        stage = str(data.get("stage") or "")
        patch["status"] = "waiting_plan_review" if stage == "plan" else "waiting_critique_review"

    elif event_type == "final":
        content = event.get("content")
        if isinstance(content, str):
            patch["current_answer"] = content
        patch["status"] = "completed"
        patch["completed"] = True

    elif event_type == "error":
        patch["status"] = "failed"
        error = event.get("error")
        if isinstance(error, Mapping):
            patch["error"] = dict(error)

    elif event_type in {"plan", "step", "observation", "task_result", "task_done", "synthesizing", "critique"}:
        patch["status"] = "running"

    await db_runs.update(pool, run_id=run_id, **patch)


async def _stream_graph_as_sse(
    graph,
    initial: ResearchState | None,
    config: dict,
    request: Request,
    abort_event: asyncio.Event,
    stream_token: str,
):
    aiter = graph.astream(initial, config, stream_mode=["custom", "updates"]).__aiter__()
    next_chunk: asyncio.Task | None = None
    disconnected = False
    waiting_for_client_close = False
    disconnect_started_at: float | None = None
    reached_terminal_event = False
    try:
        while True:
            if waiting_for_client_close:
                if await request.is_disconnected():
                    break
                yield heartbeat()
                await asyncio.sleep(SSE_REVIEW_HEARTBEAT_INTERVAL)
                continue
            if abort_event.is_set():
                disconnected = True
                break
            if await request.is_disconnected():
                now = time.monotonic()
                if disconnect_started_at is None:
                    disconnect_started_at = now
                    logger.info(
                        "SSE disconnect detected; entering grace period",
                        extra={
                            "thread_id": config.get("configurable", {}).get("thread_id"),
                            "grace_s": SSE_DISCONNECT_GRACE,
                        },
                    )
                elif now - disconnect_started_at >= SSE_DISCONNECT_GRACE:
                    abort_event.set()
                    disconnected = True
                    logger.info(
                        "SSE disconnect grace expired; aborting run",
                        extra={"thread_id": config.get("configurable", {}).get("thread_id")},
                    )
                    break
                await asyncio.sleep(0.25)
                continue
            disconnect_started_at = None
            if next_chunk is None:
                next_chunk = asyncio.create_task(aiter.__anext__())
            abort_wait = asyncio.create_task(abort_event.wait())
            try:
                done, pending = await asyncio.wait(
                    {next_chunk, abort_wait},
                    timeout=SSE_HEARTBEAT_INTERVAL,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_wait in done and abort_event.is_set():
                    disconnected = True
                    break
                if next_chunk not in done:
                    yield heartbeat()
                    continue
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    if disconnected or abort_event.is_set() or await request.is_disconnected():
                        disconnected = True
                        break
                    raise
                next_chunk = None
                event = _normalize_stream_chunk(chunk)
                if event is not None:
                    run_id = event.get("run_id")
                    if isinstance(run_id, str) and run_id:
                        await _persist_run_event(run_id, event)
                    yield encode_sse(event)
                    if event.get("type") in {"final", "error"}:
                        reached_terminal_event = True
                        break
                    if event.get("type") == "human_review":
                        waiting_for_client_close = True
                        continue
            finally:
                if not abort_wait.done():
                    abort_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await abort_wait
    except asyncio.CancelledError:
        confirmed_disconnect = abort_event.is_set()
        if not confirmed_disconnect:
            with contextlib.suppress(Exception):
                confirmed_disconnect = await request.is_disconnected()
        disconnected = disconnected or confirmed_disconnect
        logger.info(
            "SSE stream cancelled",
            extra={
                "thread_id": config.get("configurable", {}).get("thread_id"),
                "confirmed_disconnect": confirmed_disconnect,
            },
        )
    except Exception as e:
        logger.exception("SSE stream failed", extra={"thread_id": config.get("configurable", {}).get("thread_id")})
        stream_error = err.exception_error(e, source="graph")
        yield encode_sse(evt_error(stream_error.message, error=stream_error))
    finally:
        if next_chunk is not None and not next_chunk.done():
            next_chunk.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await next_chunk
        with contextlib.suppress(RuntimeError):
            await aiter.aclose()
        run_id = config.get("configurable", {}).get("thread_id")
        is_current_stream = isinstance(run_id, str) and run_id and _is_active_stream_token(request.app, run_id, stream_token)
        if disconnected and not reached_terminal_event:
            if isinstance(run_id, str) and run_id:
                if is_current_stream:
                    await _persist_run_disconnect(run_id=run_id)
                    _clear_abort_event(request.app, run_id)
        if isinstance(run_id, str) and run_id:
            _clear_active_stream_token(request.app, run_id, stream_token)


@router.get("/healthz")
async def healthz(request: Request):
    ready = hasattr(request.app.state, "graph")
    if not ready:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok", "uptime_s": round(time.time() - _START_TIME)}


@router.get("/guest")
async def guest(request: Request):
    guest_id = _get_guest_id(request) or _new_guest_id()
    response = JSONResponse({"guest_id": guest_id})
    _set_guest_cookie(response, guest_id)
    return response


@router.post("/runs/{run_id}/abort")
async def abort_run(run_id: str, request: Request):
    guest_id = _get_guest_id(request)
    pool = await db_pool.get()
    if pool is not None and guest_id:
        row = await db_runs.get(pool, run_id=run_id, guest_id=guest_id)
        if row is None:
            return JSONResponse({"error": "run not found"}, status_code=404)
    abort_event = _get_abort_event(request.app, run_id)
    abort_event.set()
    await _persist_run_disconnect(run_id=run_id)
    return {"run_id": run_id, "status": "disconnecting"}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, request: Request):
    guest_id = _get_guest_id(request)
    if not guest_id:
        return JSONResponse({"error": "guest session not found"}, status_code=404)

    pool = await db_pool.get()
    if pool is None:
        return JSONResponse({"error": "database not available"}, status_code=503)

    row = await db_runs.get(pool, run_id=run_id, guest_id=guest_id)
    if row is None:
        return JSONResponse({"error": "run not found"}, status_code=404)

    await _delete_run_artifacts(request, run_id)
    deleted = await db_runs.delete(pool, run_id=run_id, guest_id=guest_id)
    if deleted is None:
        return JSONResponse({"error": "run not found"}, status_code=404)
    return {"run_id": run_id, "deleted": True}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    run_id = str(uuid.uuid4())
    config = {"configurable": {"factory": request.app.state.factory, "thread_id": run_id}}
    final = ""
    initial = _initial_state(body.query, run_id)
    initial["human_in_loop"] = False
    async for chunk in request.app.state.graph.astream(
        initial, config, stream_mode=["custom", "updates"]
    ):
        chunk = _normalize_stream_chunk(chunk)
        if isinstance(chunk, dict) and chunk.get("type") == "final":
            final = chunk.get("content", "")
    return ChatResponse(answer=final)


@router.get("/runs/{run_id}/checkpoint")
async def latest_checkpoint(run_id: str, request: Request, full: bool = False):
    checkpoint = await request.app.state.checkpointer.aget_tuple(
        {"configurable": {"thread_id": run_id}}
    )
    if checkpoint is None:
        return JSONResponse({"error": "checkpoint not found"}, status_code=404)
    return _with_full_checkpoint(_checkpoint_to_dict(run_id, checkpoint), checkpoint, full=full)


@router.get("/runs")
async def list_runs(request: Request, limit: int = 20):
    guest_id = _get_guest_id(request)
    if not guest_id:
        return {"guest_id": None, "count": 0, "runs": []}
    pool = await db_pool.get()
    if pool is None:
        return {"guest_id": guest_id, "count": 0, "runs": []}
    rows = await db_runs.list_by_guest(pool, guest_id=guest_id, limit=max(1, min(limit, 100)))
    return {
        "guest_id": guest_id,
        "count": len(rows),
        "runs": [_run_to_dict(row) for row in rows],
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request):
    guest_id = _get_guest_id(request)
    if not guest_id:
        return JSONResponse({"error": "guest session not found"}, status_code=404)
    pool = await db_pool.get()
    if pool is None:
        return JSONResponse({"error": "database not available"}, status_code=503)
    row = await db_runs.get(pool, run_id=run_id, guest_id=guest_id)
    if row is None:
        return JSONResponse({"error": "run not found"}, status_code=404)

    checkpoint = await request.app.state.checkpointer.aget_tuple(
        {"configurable": {"thread_id": run_id}}
    )
    payload = _run_to_dict(row, include_answer=True)
    payload["has_checkpoint"] = checkpoint is not None
    if checkpoint is not None:
        payload["latest_checkpoint"] = _checkpoint_to_dict(run_id, checkpoint)
    return payload


@router.get("/runs/{run_id}/checkpoints")
async def list_checkpoints(
    run_id: str,
    request: Request,
    limit: int | None = None,
    full: bool = False,
):
    checkpoints = []
    async for checkpoint in request.app.state.checkpointer.alist(
        {"configurable": {"thread_id": run_id}},
        limit=limit,
    ):
        checkpoints.append(_with_full_checkpoint(_checkpoint_to_dict(run_id, checkpoint), checkpoint, full=full))
    return {"run_id": run_id, "count": len(checkpoints), "checkpoints": checkpoints}


@router.get("/runs/{run_id}/checkpoints/{checkpoint_id}")
async def get_checkpoint(run_id: str, checkpoint_id: str, request: Request, full: bool = False):
    checkpoint = await request.app.state.checkpointer.aget_tuple(
        {"configurable": {"thread_id": run_id, "checkpoint_id": checkpoint_id}}
    )
    if checkpoint is None:
        return JSONResponse({"error": "checkpoint not found"}, status_code=404)
    return _with_full_checkpoint(_checkpoint_to_dict(run_id, checkpoint), checkpoint, full=full)


@router.get("/stream")
async def stream(
    request: Request,
    query: str | None = None,
    run_id: str | None = None,
    resume: bool = False,
    action: str | None = None,
    feedback: str | None = None,
):
    run_id = run_id or str(uuid.uuid4())
    guest_id = _get_guest_id(request) or _new_guest_id()
    abort_event = _get_abort_event(request.app, run_id)
    abort_event.clear()
    stream_token = str(uuid.uuid4())
    _set_active_stream_token(request.app, run_id, stream_token)
    config = {"configurable": {"factory": request.app.state.factory, "thread_id": run_id}}
    initial: ResearchState | Command | None

    if resume:
        checkpoint = await request.app.state.checkpointer.aget_tuple(
            {"configurable": {"thread_id": run_id}}
        )
        if checkpoint is None:
            response = JSONResponse({"error": "checkpoint not found"}, status_code=404)
            _set_guest_cookie(response, guest_id)
            return response
        if action:
            initial = Command(resume={"action": action, "feedback": (feedback or "").strip()})
        else:
            initial = None
        await _persist_run_resume(run_id=run_id, guest_id=guest_id)
    else:
        if not query:
            response = JSONResponse({"error": "query is required unless resume=true"}, status_code=400)
            _set_guest_cookie(response, guest_id)
            return response
        initial = _initial_state(query, run_id)
        await _persist_run_start(run_id=run_id, guest_id=guest_id, query=query)

    response = StreamingResponse(
        _stream_graph_as_sse(request.app.state.graph, initial, config, request, abort_event, stream_token),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Run-ID": run_id},
    )
    response.set_cookie(
        _GUEST_COOKIE,
        guest_id,
        max_age=60 * 60 * 24 * 180,
        samesite="lax",
        httponly=False,
    )
    return response


def _normalize_stream_chunk(chunk):
    custom = _unwrap_custom_event(chunk)
    if custom is not chunk:
        return custom

    if (
        isinstance(chunk, (list, tuple))
        and len(chunk) == 2
        and chunk[0] == "updates"
        and isinstance(chunk[1], dict)
    ):
        interrupt_items = chunk[1].get("__interrupt__") or ()
        if interrupt_items:
            interrupt = interrupt_items[0]
            value = getattr(interrupt, "value", {}) or {}
            interrupt_id = getattr(interrupt, "id", None)
            run_id = value.get("run_id")
            return evt_human_review(
                stage=value.get("stage", "human"),
                prompt=value.get("prompt", ""),
                options=value.get("options", []),
                payload=value,
                run_id=run_id,
                interrupt_id=interrupt_id,
            )
        return None

    return None
