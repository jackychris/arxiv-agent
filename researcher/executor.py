# researcher/executor.py
import asyncio
import json
import logging
import re
import uuid

from a2a.helpers.proto_helpers import new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import Task, TaskState, TaskStatus

import llm
import memory.long_term as lt
from config import MAX_STEPS  # noqa: F401 (MAX_STEPS used in _parse_mission)
from mcp_client import MCPClient
from memory.short_term import ShortTermMemory
from prompts import TOOL_REFLECT_PROMPT, build_memory_hint, build_system_prompt
from researcher import errors as research_errors
from researcher import events as research_events
from researcher.reflection import build_reflection_history
from runtime import trace

logger = logging.getLogger(__name__)

_STEPS_PREFIX_RE = re.compile(r"^\[steps=(\d+)\]\s*")
_URL_RE = re.compile(r"https?://[^\s<>)\\\"']+")


def _parse_mission(raw: str) -> tuple[int, str]:
    """Return (max_steps, clean_mission). Falls back to config MAX_STEPS if no prefix."""
    m = _STEPS_PREFIX_RE.match(raw)
    if m:
        return int(m.group(1)), raw[m.end():]
    return MAX_STEPS, raw


_OBS_LIMIT_DEFAULT = 6000
_OBS_LIMIT: dict[str, int] = {
    "read_arxiv_paper": 12000,
    "read_semantic_paper": 12000,
    "read_openalex_paper": 12000,
    "read_dblp_paper": 12000,
    "download_arxiv": 12000,
    "download_semantic": 12000,
    "get_file_contents": 10000,
    "tavily_extract": 10000,
}


def _iter_items(data) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("results", "result", "papers", "items", "data"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [data]


def _first_text(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _paper_citation(item: dict) -> dict | None:
    title = _first_text(item, ("title", "name"))
    doi = _first_text(item, ("doi", "DOI"))
    if doi:
        return {"key": f"doi:{doi.lower()}", "title": title[:100], "cited": False}
    source = _first_text(item, ("source",)).lower()
    url = _first_text(item, ("url", "pdf_url"))
    arxiv_id = _first_text(item, ("arxiv_id", "arxivId", "arxiv", "id"))
    if not arxiv_id and source == "arxiv":
        arxiv_id = _first_text(item, ("paper_id", "id"))
    if not arxiv_id and "arxiv.org/" in url:
        arxiv_id = _first_text(item, ("paper_id", "id"))
    if arxiv_id:
        return {"key": f"arxiv:{arxiv_id}", "title": title[:100], "cited": False}
    semantic_id = _first_text(item, ("paperId", "semantic_scholar_id"))
    if not semantic_id and source in ("semantic", "semantic_scholar", "semanticscholar"):
        semantic_id = _first_text(item, ("paper_id", "id"))
    if semantic_id:
        return {"key": f"semantic:{semantic_id}", "title": title[:100], "cited": False}
    url = _first_text(item, ("url", "pdf_url", "openAccessPdf", "externalIds"))
    if url:
        return {"key": f"url:{url}", "title": title[:100] or url[:100], "cited": False}
    return None


def _github_citation(item: dict) -> dict | None:
    repo = _first_text(item, ("full_name", "fullName", "nameWithOwner"))
    repository = item.get("repository")
    if not repo and isinstance(repository, dict):
        repo = _first_text(repository, ("full_name", "fullName", "nameWithOwner"))
    if repo:
        path = _first_text(item, ("path", "file_path"))
        key = f"github:{repo}:{path}" if path else f"github:{repo}"
        return {"key": key, "title": path or repo, "cited": False}
    url = _first_text(item, ("html_url", "url"))
    if url and "github.com/" in url:
        return {"key": f"url:{url}", "title": url[:100], "cited": False}
    return None


def _url_citations(data) -> list[dict]:
    out = []
    if isinstance(data, str):
        seen = set()
        for url in _URL_RE.findall(data):
            url = url.rstrip(".,;:")
            if url in seen:
                continue
            seen.add(url)
            out.append({"key": f"url:{url}", "title": url[:100], "cited": False})
        return out
    for item in _iter_items(data):
        url = _first_text(item, ("url", "raw_content_url"))
        if url:
            out.append({"key": f"url:{url}", "title": _first_text(item, ("title", "name"))[:100] or url[:100], "cited": False})
    return out


def _citations_from_observation(tool: str, observation: str) -> list[dict]:
    """Extract structured citations from a successful tool observation."""
    try:
        envelope = json.loads(observation)
    except json.JSONDecodeError:
        return []
    if not envelope.get("ok"):
        return []
    data = envelope.get("data")
    if data is None:
        return []

    if tool in (
        "search_semantic",
        "search_arxiv",
        "search_openalex",
        "search_crossref",
        "get_crossref_paper_by_doi",
        "search_dblp",
        "download_arxiv",
        "download_semantic",
        "read_arxiv_paper",
        "read_semantic_paper",
        "read_openalex_paper",
        "read_dblp_paper",
    ):
        citations = [_paper_citation(item) for item in _iter_items(data)]
        return [citation for citation in citations[:5] if citation is not None]

    if tool in ("search_repositories", "search_code", "get_file_contents"):
        citations = [_github_citation(item) for item in _iter_items(data)]
        return [citation for citation in citations[:5] if citation is not None]

    if tool in ("tavily_search", "tavily_extract"):
        return _url_citations(data)[:5]

    return []


def _parse(text: str) -> dict | None:
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
        return None
    except json.JSONDecodeError:
        return None


class ResearchAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id: str = context.task_id or ""
        context_id: str = context.context_id or ""
        task_max_steps, mission = _parse_mission(context.get_user_input())

        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        memories = await lt.get_all()
        tool_memories = {k: v for k, v in memories.items() if k != "orchestrator"}
        memory_hint = build_memory_hint(tool_memories)

        final_answer: str | None = None
        _cancelled = False
        memory = ShortTermMemory()
        citations: list[dict] = []
        trace.record("agent_task_start", run_id=context_id, task_id=task_id, content=mission)

        try:
            logger.info("Task %s entering MCPClient", task_id)
            async with MCPClient() as client:
                logger.info("Task %s MCPClient ready", task_id)
                memory.add("system", build_system_prompt(client.get_tools_description()))
                if memory_hint:
                    memory.add("system", f"Past tool experience:\n{memory_hint}")
                memory.add("user", mission)

                for step in range(task_max_steps):
                    logger.info("Task %s step %d/%d starting", task_id, step + 1, task_max_steps)
                    remaining = task_max_steps - step
                    if remaining <= 2:
                        step_hint = f"{remaining} step(s) remaining. Output {{\"done\": true}} if you have enough information, otherwise use your last tool call."
                    else:
                        step_hint = f"{remaining} steps remaining."
                    extras = [{"role": "system", "content": step_hint}]
                    temp = memory.get() + extras

                    try:
                        logger.info("Task %s step %d calling llm.chat", task_id, step + 1)
                        response = await llm.chat(temp)
                        logger.info("Task %s step %d llm.chat returned", task_id, step + 1)
                    except Exception as e:
                        memory.add("user", research_errors.llm_error(e))
                        continue
                    memory.add("assistant", response)

                    data = _parse(response)
                    if data is None:
                        memory.add("user", research_errors.invalid_json())
                        continue

                    if data.get("done"):
                        break

                    action = data.get("action", "")
                    action_input = data.get("action_input", {})
                    thought = data.get("thought", "")

                    if not action:
                        memory.add("user", research_errors.missing_action())
                        continue

                    trace.record(
                        "agent_step",
                        run_id=context_id,
                        task_id=task_id,
                        step=step + 1,
                        tool=action,
                        content=thought[:500],
                        data={"tool_input": action_input},
                    )
                    step_summary = research_events.step_status(
                        step + 1, thought, action, action_input
                    )
                    logger.info("Task %s step %d publishing status for tool %s", task_id, step + 1, action)
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=updater.new_agent_message([new_text_part(step_summary)]),
                    )

                    tool_call_id = str(uuid.uuid4())
                    started = trace.monotonic_ms()
                    try:
                        logger.info("Task %s step %d executing tool %s", task_id, step + 1, action)
                        observation = await client.execute_tool(action, **action_input)
                        logger.info("Task %s step %d tool %s returned", task_id, step + 1, action)
                    except Exception as e:
                        observation = research_errors.tool_error(e)
                    ok, error_code = trace.summarize_tool_result(observation)
                    trace.record(
                        "tool_result",
                        run_id=context_id,
                        task_id=task_id,
                        step=step + 1,
                        tool=action,
                        tool_call_id=tool_call_id,
                        latency_ms=trace.elapsed_ms(started),
                        ok=ok,
                        error_code=error_code,
                        content=observation[:500],
                    )

                    new_citations = _citations_from_observation(action, observation)
                    citations.extend(new_citations)
                    _obs_limit = _OBS_LIMIT.get(action, _OBS_LIMIT_DEFAULT)
                    obs_truncated = observation[:_obs_limit] + (
                        "\n[truncated]" if len(observation) > _obs_limit else ""
                    )
                    memory.add("user", f"Observation: {obs_truncated}")

                    logger.info("Task %s step %d publishing observation for tool %s", task_id, step + 1, action)
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=updater.new_agent_message(
                            [
                                new_text_part(
                                    research_events.observation_status(
                                        observation, step + 1, action
                                    )
                                )
                            ]
                        ),
                    )

                logger.info("Task %s building final artifact", task_id)
                final_answer = json.dumps(
                    {"history": build_reflection_history(memory), "citations": citations},
                    ensure_ascii=False,
                )
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError):
                _cancelled = True
            if final_answer is None:
                final_answer = research_errors.task_interrupted(e)
            logger.warning(
                "Task %s interrupted by %s: %s", task_id, type(e).__name__, e, exc_info=True
            )

        if final_answer is None:
            final_answer = research_errors.no_answer()
        trace.record(
            "agent_task_result",
            run_id=context_id,
            task_id=task_id,
            ok=not research_errors.is_failure_text(final_answer),
            content=final_answer[:500],
        )

        try:
            await updater.add_artifact(
                parts=[new_text_part(final_answer)],
                name="final_answer",
            )
            await updater.complete()
        except BaseException as e:
            logger.warning(
                "Task %s failed while publishing final artifact: %s: %s",
                task_id,
                type(e).__name__,
                e,
                exc_info=True,
            )

        if _cancelled:
            raise asyncio.CancelledError()

        trace.record("agent_task_done", run_id=context_id, task_id=task_id)
        await self._reflect(mission, memory)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.cancel()

    async def _reflect(self, mission: str, memory: ShortTermMemory) -> None:
        history = build_reflection_history(memory)
        prompt = TOOL_REFLECT_PROMPT.format(mission=mission, history=history)
        try:
            response = await llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(response)
            for tool, reflection in data.items():
                if reflection and reflection != "null":
                    await lt.add(tool, reflection)
        except Exception:
            logger.warning("Task %s reflection failed", mission[:80], exc_info=True)
