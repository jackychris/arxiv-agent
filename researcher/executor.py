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


def _parse_mission(raw: str) -> tuple[int, str]:
    """Return (max_steps, clean_mission). Falls back to config MAX_STEPS if no prefix."""
    m = _STEPS_PREFIX_RE.match(raw)
    if m:
        return int(m.group(1)), raw[m.end():]
    return MAX_STEPS, raw


_OBS_LIMIT_DEFAULT = 6000
_OBS_LIMIT: dict[str, int] = {
    "fetch_url": 10000,
    "get_repo_readme": 10000,
    "get_paper_detail": 10000,
}


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

    if tool in ("search_semantic_scholar", "search_arxiv"):
        items = data if isinstance(data, list) else []
        out = []
        for p in items[:5]:
            if not isinstance(p, dict):
                continue
            if p.get("id"):
                out.append({"key": f"arxiv:{p['id']}", "title": p.get("title", "")[:100], "cited": False})
            elif p.get("arxiv_id"):
                out.append({"key": f"arxiv:{p['arxiv_id']}", "title": p.get("title", "")[:100], "cited": False})
            elif p.get("pdf_url"):
                out.append({"key": f"url:{p['pdf_url']}", "title": p.get("title", "")[:100], "cited": False})
        return out

    if tool == "web_search":
        items = data if isinstance(data, list) else []
        out = []
        for r in items[:5]:
            if isinstance(r, dict) and r.get("url"):
                out.append({"key": f"url:{r['url']}", "title": r.get("title", "")[:100], "cited": False})
        return out

    if tool == "get_paper_content" and isinstance(data, dict) and data.get("id"):
        return [{"key": f"arxiv:{data['id']}", "title": data.get("title", "")[:100], "cited": False}]

    if tool in ("search_repos", "search_code"):
        items = data if isinstance(data, list) else []
        return [
            {"key": f"github:{r['full_name']}", "title": r.get("full_name", ""), "cited": False}
            for r in items[:3]
            if isinstance(r, dict) and r.get("full_name")
        ]

    if tool == "get_repo_readme" and isinstance(data, dict) and data.get("full_name"):
        return [{"key": f"github:{data['full_name']}", "title": data.get("full_name", ""), "cited": False}]

    if tool == "fetch_url" and isinstance(data, dict) and data.get("url"):
        return [{"key": f"url:{data['url']}", "title": data.get("url", ""), "cited": False}]

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
