# orchestrator/agent.py
import asyncio
import json
import logging
import uuid

import httpx
import llm

logger = logging.getLogger(__name__)
import memory.long_term as lt
from prompts import ORCHESTRATE_PROMPT, SYNTHESIZE_PROMPT, ORCHESTRATOR_REFLECT_PROMPT, build_memory_hint

from orchestrator import events as evt
from orchestrator import errors as err
from a2a.client.client_factory import ClientFactory, ClientConfig
from a2a.helpers import get_artifact_text, get_message_text
from a2a.helpers.proto_helpers import new_text_message
from a2a.types.a2a_pb2 import SendMessageRequest, TaskState

from runtime import run_context
from runtime import trace
from config import A2A_CLIENT_TIMEOUT, PORT_RESEARCH_AGENT

RESEARCH_AGENT_URL = f"http://localhost:{PORT_RESEARCH_AGENT}"


class OrchestratorAgent:
    async def _plan(self, query: str) -> list[dict]:
        orch_memories = lt.get("orchestrator")
        memory_hint = build_memory_hint({"orchestrator": orch_memories})
        extra = f"\n\nPast planning experience:\n{memory_hint}" if memory_hint else ""
        prompt = ORCHESTRATE_PROMPT.format(query=query) + extra
        response = await llm.chat([{"role": "user", "content": prompt}])
        return json.loads(response)["tasks"]

    async def _synthesize(self, query: str, subagent_results: list[str]) -> str:
        results_text = "\n\n".join(
            f"Subagent {i + 1}:\n{r}" for i, r in enumerate(subagent_results)
        )
        prompt = SYNTHESIZE_PROMPT.format(query=query, results=results_text)
        return await llm.complete([{"role": "user", "content": prompt}])

    async def _reflect(self, query: str, tasks: list[dict], subagent_results: list[str]) -> None:
        outcomes = "\n".join(
            f"Subagent {i + 1} ({t['mission'][:80]}): {'failed' if err.is_failure_text(r) else 'succeeded'}"
            for i, (t, r) in enumerate(zip(tasks, subagent_results))
        )
        prompt = ORCHESTRATOR_REFLECT_PROMPT.format(
            query=query,
            tasks="\n".join(f"- {t['mission']}" for t in tasks),
            outcomes=outcomes,
        )
        try:
            reflection = await llm.complete([{"role": "user", "content": prompt}])
            if reflection.strip():
                lt.add("orchestrator", reflection.strip())
        except Exception:
            logger.warning("Orchestrator reflection failed", exc_info=True)

    async def _run_task(
        self,
        factory: ClientFactory,
        mission: str,
        context_id: str,
        task_label: str,
        queue: asyncio.Queue,
    ) -> str:
        final_answer: str | None = None
        final_error: err.AgentError | None = None
        result_sent = False
        trace.record("task_start", run_id=context_id, task_id=task_label, content=mission)
        try:
            client = await factory.create_from_url(RESEARCH_AGENT_URL)
            msg = new_text_message(mission, context_id=context_id)
            req = SendMessageRequest(message=msg)

            async for event in client.send_message(req):
                if event.HasField("artifact_update"):
                    final_answer = get_artifact_text(event.artifact_update.artifact)
                    if final_answer:
                        trace.record(
                            "task_result",
                            run_id=context_id,
                            task_id=task_label,
                            ok=True,
                            content=final_answer[:500],
                        )
                        await queue.put(evt.task_result(task_label, final_answer, run_id=context_id))
                        result_sent = True
                elif event.HasField("status_update"):
                    state = event.status_update.status.state
                    if state == TaskState.TASK_STATE_WORKING:
                        text = get_message_text(event.status_update.status.message)
                        if text:
                            try:
                                await queue.put(evt.status_text(task_label, text, run_id=context_id))
                            except ValueError as e:
                                status_error = err.exception_error(e, source="orchestrator", code="A2A_STATUS_FORMAT_ERROR")
                                await queue.put(evt.observation(
                                    task_label,
                                    status_error.message,
                                    run_id=context_id,
                                    data={"raw_status": text},
                                ))
                    elif state in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_CANCELED):
                        text = get_message_text(event.status_update.status.message)
                        state_name = TaskState.Name(state).removeprefix("TASK_STATE_").lower()
                        reason = text or "task ended before producing an artifact"
                        final_error = err.task_state_error(state_name, reason)
                        final_answer = err.failed_text(final_error)
                        break
        except Exception as e:
            final_error = err.exception_error(e, source="orchestrator", code="A2A_NETWORK_ERROR")
            final_answer = err.failed_text(final_error)
        finally:
            if not result_sent:
                if final_answer is None:
                    final_error = err.no_result()
                    final_answer = err.failed_text(final_error)
                trace.record(
                    "task_result",
                    run_id=context_id,
                    task_id=task_label,
                    ok=final_error is None,
                    error_code=final_error.code if final_error else None,
                    content=final_answer[:500],
                )
                await queue.put(evt.task_result(task_label, final_answer, run_id=context_id, error=final_error))
            trace.record("task_done", run_id=context_id, task_id=task_label)
            await queue.put(evt.task_done(task_label, run_id=context_id))

        return final_answer or err.failed_text(err.no_result())

    async def run_streaming(self, query: str, queue: asyncio.Queue) -> None:
        context_id = str(uuid.uuid4())
        try:
            trace.record("run_start", run_id=context_id, content=query)
            tasks = await self._plan(query)
            run_context.clear(context_id)
            trace.record(
                "plan_created",
                run_id=context_id,
                data={"tasks": [{"id": t["id"], "mission": t["mission"]} for t in tasks]},
            )

            await queue.put(evt.plan(
                [{"id": t["id"], "mission": t["mission"]} for t in tasks],
                run_id=context_id,
            ))

            async with httpx.AsyncClient(timeout=A2A_CLIENT_TIMEOUT) as http:
                factory = ClientFactory(ClientConfig(httpx_client=http))

                coros = [
                    self._run_task(factory, t["mission"], context_id, t["id"], queue)
                    for t in tasks
                ]
                raw_results = await asyncio.gather(*coros, return_exceptions=True)

            run_context.clear(context_id)

            subagent_results = [
                r if not isinstance(r, Exception) else err.failed_text(err.exception_error(r, "orchestrator"))
                for r in raw_results
            ]

            any_failed = any(err.is_failure_text(r) for r in subagent_results)
            if any_failed:
                await self._reflect(query, tasks, subagent_results)

            if len(subagent_results) == 1:
                final = subagent_results[0]
            else:
                trace.record("synthesis_start", run_id=context_id)
                await queue.put(evt.synthesizing(run_id=context_id))
                final = await self._synthesize(query, subagent_results)

            trace.record("run_final", run_id=context_id, content=final[:1000])
            await queue.put(evt.final(final, run_id=context_id))

        except Exception as e:
            stream_error = err.exception_error(e, source="orchestrator")
            trace.record(
                "run_error",
                run_id=context_id,
                error_code=stream_error.code,
                content=stream_error.message,
            )
            await queue.put(evt.error(stream_error.message, error=stream_error))
        finally:
            await queue.put(None)

    async def run(self, query: str, verbose: bool = False) -> str:
        queue: asyncio.Queue = asyncio.Queue()
        final = ""

        async def drain():
            nonlocal final
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event["type"] == "final":
                    final = event["content"]
                elif verbose:
                    if event["type"] == "plan":
                        logger.info("Orchestrator: %d task(s)", len(event["tasks"]))
                        for t in event["tasks"]:
                            logger.info("  [%s] %s", t["id"], t["mission"])
                    elif event["type"] == "step":
                        logger.info("  [Task %s] %s", event["task_id"], event["content"])
                    elif event["type"] == "synthesizing":
                        logger.info("Synthesizing results")

        await asyncio.gather(self.run_streaming(query, queue), drain())
        return final
