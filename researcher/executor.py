# researcher/executor.py
import asyncio
import json
import logging
import uuid

from a2a.helpers.proto_helpers import new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import Task, TaskState, TaskStatus

import llm
import memory.long_term as lt
from config import ENABLE_RUN_CONTEXT, MAX_STEPS
from mcp_client import MCPClient
from memory.short_term import ShortTermMemory
from prompts import TOOL_REFLECT_PROMPT, build_memory_hint, build_system_prompt
from researcher import errors as research_errors
from researcher import events as research_events
from researcher.reflection import build_reflection_history
from runtime import run_context, trace

logger = logging.getLogger(__name__)

_OBS_LIMIT_DEFAULT = 6000
_OBS_LIMIT: dict[str, int] = {
    "fetch_url": 10000,
    "get_repo_readme": 10000,
    "get_paper_detail": 10000,
}


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
        mission = context.get_user_input()

        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        memories = lt.get_all()
        tool_memories = {k: v for k, v in memories.items() if k != "orchestrator"}
        memory_hint = build_memory_hint(tool_memories)

        final_answer: str | None = None
        _cancelled = False
        memory = ShortTermMemory()
        trace.record("task_start", run_id=context_id, task_id=task_id, content=mission)

        try:
            async with MCPClient() as client:
                memory.add("system", build_system_prompt(client.get_tools_description()))
                if memory_hint:
                    memory.add("system", f"Past tool experience:\n{memory_hint}")
                memory.add("user", mission)

                for step in range(MAX_STEPS):
                    remaining = MAX_STEPS - step
                    if remaining == 1:
                        step_hint = "1 step remaining. You must now give your final_answer. Do not call any tools."
                    elif remaining == 2:
                        step_hint = "2 steps remaining. This is your last chance to call a tool. Use it wisely."
                    else:
                        step_hint = f"{remaining} steps remaining."
                    others = (
                        run_context.get_others(context_id, task_id) if ENABLE_RUN_CONTEXT else ""
                    )
                    extras = []
                    if others:
                        extras.append({"role": "system", "content": others})
                    extras.append({"role": "system", "content": step_hint})
                    temp = memory.get() + extras

                    try:
                        response = await llm.chat(temp)
                    except Exception as e:
                        memory.add("user", research_errors.llm_error(e))
                        continue
                    memory.add("assistant", response)

                    data = _parse(response)
                    if data is None:
                        memory.add("user", research_errors.invalid_json())
                        continue

                    if "final_answer" in data:
                        final_answer = str(data["final_answer"])
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
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING,
                        message=updater.new_agent_message([new_text_part(step_summary)]),
                    )

                    tool_call_id = str(uuid.uuid4())
                    started = trace.monotonic_ms()
                    try:
                        observation = await client.execute_tool(action, **action_input)
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

                    if ENABLE_RUN_CONTEXT:
                        run_context.publish(context_id, task_id, step + 1, observation)
                    _obs_limit = _OBS_LIMIT.get(action, _OBS_LIMIT_DEFAULT)
                    obs_truncated = observation[:_obs_limit] + (
                        "\n[truncated]" if len(observation) > _obs_limit else ""
                    )
                    memory.add("user", f"Observation: {obs_truncated}")

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

                if final_answer is None:
                    memory.add(
                        "system",
                        "You have reached the maximum number of steps. Based on what you have found so far, give your best final_answer now. Do not call any tools.",
                    )
                    try:
                        response = await llm.chat(memory.get())
                        memory.add("assistant", response)
                        data = _parse(response)
                        if data and "final_answer" in data:
                            final_answer = f"[Max steps reached] {data['final_answer'] or response}"
                        else:
                            final_answer = response
                    except Exception as e:
                        final_answer = research_errors.final_generation_error(e)
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
            "task_result",
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

        trace.record("task_done", run_id=context_id, task_id=task_id)
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
                    lt.add(tool, reflection)
        except Exception:
            logger.warning("Task %s reflection failed", mission[:80], exc_info=True)
