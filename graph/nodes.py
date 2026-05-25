# graph/nodes.py
import asyncio
import contextlib
import json
import logging
import uuid
from datetime import UTC, datetime

from a2a.helpers import get_artifact_text, get_message_text
from a2a.helpers.proto_helpers import new_text_message
from a2a.types.a2a_pb2 import CancelTaskRequest, SendMessageRequest
from a2a.types.a2a_pb2 import TaskState as A2ATaskState
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter, interrupt

import llm
import memory.long_term as lt
from config import (
    CRITIC_MAX_MISSIONS,
    EFFORT_STEPS,
    MAX_CRITIC_ROUNDS,
    MAX_STEPS,
    PORT_RESEARCH_AGENT,
)
from graph import errors as err
from graph import events as evt
from graph.synthesizer import strip_sources_section, synthesize_with_evaluation
from graph.state import ResearchState, TaskState
from prompts import (
    CONTINUE_GUIDANCE_PROMPT,
    CRITIC_PROMPT,
    ORCHESTRATE_FOLLOWUP_PROMPT,
    ORCHESTRATE_PROMPT,
    ORCHESTRATOR_REFLECT_PROMPT,
    QUERY_REWRITE_PROMPT,
    build_memory_hint,
)
from runtime import run_context, trace
from researcher.server import get_agent_card

logger = logging.getLogger(__name__)

SOURCE = "graph"
RESEARCH_AGENT_URL = f"http://127.0.0.1:{PORT_RESEARCH_AGENT}"
CITATIONS_CONTEXT_LIMIT = 8
CITATION_TEXT_LIMIT = 180
DEFAULT_CONTINUE_GUIDANCE = "Continue researching to verify remaining uncertainties and materially strengthen weak parts of the answer."


def _short(value: str | None, limit: int = CITATION_TEXT_LIMIT) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _format_known_citations(citations: list[dict], limit: int = CITATIONS_CONTEXT_LIMIT) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations[-limit:]:
        key = c.get("key", "source")
        title = _short(c.get("title") or key, 120)
        cited = "cited" if c.get("cited") else "uncited"
        ref_num = c.get("ref_num")
        label = f"#{ref_num} [{key}]" if ref_num is not None else f"[{key}]"
        lines.append(f"- {label} {title} ({cited})".strip())
    return "\n".join(lines)


def _preview_text(value: str | None, limit: int = 240) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _summarize_histories(histories: list[str], limit: int = 3) -> list[str]:
    return [_preview_text(history, 180) for history in histories[:limit]]


def _summarize_citations(citations: list[dict], limit: int = 8) -> list[dict]:
    items: list[dict] = []
    for citation in citations[:limit]:
        items.append(
            {
                "ref_num": citation.get("ref_num"),
                "key": citation.get("key"),
                "title": _preview_text(str(citation.get("title") or ""), 120),
                "cited": bool(citation.get("cited")),
            }
        )
    return items


def _parse_findings(raw_results: list[str]) -> tuple[list[str], list[dict]]:
    histories: list[str] = []
    citations: list[dict] = []
    seen: set[str] = set()
    for raw in raw_results:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "history" in data:
                histories.append(data["history"])
                for c in data.get("citations", []):
                    key = c.get("key") if isinstance(c, dict) else None
                    if key and key not in seen:
                        seen.add(key)
                        c.setdefault("cited", False)
                        citations.append(c)
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        histories.append(raw)
    return histories, citations


def _normalize_human_feedback(value: object) -> str:
    return str(value or "").strip()


def _merge_guidance(base: str, extra: str) -> str:
    base = base.strip()
    extra = extra.strip()
    if base and extra:
        return f"{base}\n\nAdditional user guidance:\n{extra}"
    return extra or base


async def _generate_continue_guidance(state: ResearchState) -> str:
    prompt = CONTINUE_GUIDANCE_PROMPT.format(
        query=state["rewritten_query"],
        answer=state.get("current_answer", ""),
    )
    try:
        guidance = (await llm.complete([{"role": "user", "content": prompt}])).strip()
        return guidance or DEFAULT_CONTINUE_GUIDANCE
    except Exception:
        logger.warning("Continue guidance generation failed, using default guidance", exc_info=True)
        return DEFAULT_CONTINUE_GUIDANCE

# ---------------------------------------------------------------------------
# init / finalize
# ---------------------------------------------------------------------------

async def init(state: ResearchState, writer: StreamWriter) -> dict:
    context_id = state.get("context_id") or str(uuid.uuid4())
    if (
        state.get("round", 0) == 0
        and not state.get("current_answer")
        and not state.get("subagent_results")
    ):
        run_context.clear(context_id)
    trace.record("run_start", run_id=context_id, content=state["query"])
    writer(
        evt.node_result(
            "init",
            run_id=context_id,
            content=state["query"],
            data={"query": state["query"], "context_id": context_id, "round": state.get("round", 0)},
        )
    )
    return {"context_id": context_id}


async def finalize(state: ResearchState, writer: StreamWriter) -> dict:
    context_id = state["context_id"]
    final = state.get("current_answer", "")
    trace.record("run_final", run_id=context_id, content=final[:1000])
    writer(
        evt.node_result(
            "finalize",
            run_id=context_id,
            content=final,
            data={"current_answer": final, "round": state.get("round", 0)},
        )
    )
    writer(evt.final(final, run_id=context_id))
    run_context.clear(context_id)
    return {}


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------

async def rewrite_query(state: ResearchState, writer: StreamWriter) -> dict:
    query = state["query"]
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    prompt = QUERY_REWRITE_PROMPT.format(query=query, today=today)
    rewritten = (await llm.complete([{"role": "user", "content": prompt}])).strip()
    rewritten = rewritten if rewritten else query
    if rewritten != query:
        trace.record("query_rewritten", run_id=state["context_id"], content=rewritten)
    writer(
        evt.node_result(
            "rewrite_query",
            run_id=state["context_id"],
            content=rewritten,
            data={
                "original_query": query,
                "rewritten_query": rewritten,
                "changed": rewritten != query,
                "round": state.get("round", 0),
            },
        )
    )
    return {"rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# plan_tasks
# ---------------------------------------------------------------------------

async def plan_tasks(state: ResearchState, writer: StreamWriter) -> dict:
    context_id = state["context_id"]
    planning_guidance = state.get("planning_guidance", "")
    critique_gaps = state.get("critique_gaps", [])
    plan_feedback = _normalize_human_feedback(state.get("plan_review_feedback"))
    known_citations = _format_known_citations(state.get("known_citations", []))

    orch_memories = await lt.get("orchestrator")
    memory_hint = build_memory_hint({"orchestrator": orch_memories})
    extra = f"\n\nPast planning experience:\n{memory_hint}" if memory_hint else ""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    citations_hint = (
        "\n\nKnown citations from earlier rounds; plan only genuinely missing work and avoid duplicate source lookups:\n"
        + known_citations
        if known_citations
        else ""
    )
    feedback_hint = (
        "\n\nUser feedback for replanning:\n" + plan_feedback
        if plan_feedback
        else ""
    )

    if planning_guidance:
        prompt = (
            ORCHESTRATE_FOLLOWUP_PROMPT.format(
                query=state["rewritten_query"],
                answer=state.get("current_answer", ""),
                gaps="\n".join(f"- {gap}" for gap in critique_gaps) or "- None provided",
                guidance=planning_guidance,
            )
            + extra
            + citations_hint
            + feedback_hint
            + f"\n\nToday's date: {today} (UTC)."
        )
        response = await llm.chat([{"role": "user", "content": prompt}])
        tasks = json.loads(response)["tasks"]
        for t in tasks:
            t["id"] = str(uuid.uuid4())
    else:
        prompt = (
            ORCHESTRATE_PROMPT.format(query=state["rewritten_query"])
            + extra
            + citations_hint
            + feedback_hint
            + f"\n\nToday's date: {today} (UTC)."
        )
        response = await llm.chat([{"role": "user", "content": prompt}])
        tasks = json.loads(response)["tasks"]
        for t in tasks:
            t["id"] = str(uuid.uuid4())

    mission_suffix = (
        "\n\nKnown citations:\n"
        + known_citations
        + "\n\nUse these as already-known sources. Avoid duplicate lookups unless you need missing details."
        if known_citations
        else ""
    )

    for t in tasks:
        steps = EFFORT_STEPS.get(t.get("effort", ""), MAX_STEPS)
        agent_mission = t["mission"]
        if mission_suffix and mission_suffix not in agent_mission:
            agent_mission += mission_suffix
        t["encoded_mission"] = f"[steps={steps}] {agent_mission}"

    trace.record(
        "plan_created",
        run_id=context_id,
        data={"round": state.get("round", 0), "tasks": [{"id": t["id"], "mission": t["mission"]} for t in tasks]},
    )
    writer(
        evt.node_result(
            "plan_tasks",
            run_id=context_id,
            content=json.dumps(tasks, ensure_ascii=False, indent=2),
            data={"round": state.get("round", 0), "tasks": tasks},
        )
    )
    writer(evt.plan([{"id": t["id"], "mission": t["mission"]} for t in tasks], run_id=context_id, round=state.get("round", 0)))

    return {
        "tasks": tasks,
        "critique_gaps": [],
        "planning_guidance": "",
        "plan_review_action": "",
        "plan_review_feedback": "",
        "critique_review_action": "",
        "critique_review_feedback": "",
    }


async def review_plan(state: ResearchState, writer: StreamWriter) -> dict:
    context_id = state["context_id"]
    if not state.get("human_in_loop", True):
        return {"plan_review_action": "continue", "plan_review_feedback": ""}

    task_preview = [
        {"id": task.get("id"), "mission": task.get("mission", ""), "effort": task.get("effort", "medium")}
        for task in state.get("tasks", [])
    ]
    payload = {
        "stage": "plan",
        "run_id": context_id,
        "round": state.get("round", 0),
        "tasks": task_preview,
        "prompt": "Review the task plan before launching research.",
        "options": [
            {"action": "continue", "label": "Continue"},
            {"action": "replan", "label": "Replan"},
            {"action": "replan_with_feedback", "label": "Replan with Feedback"},
        ],
    }
    writer(
        evt.node_result(
            "review_plan",
            run_id=context_id,
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            status="waiting",
        )
    )
    decision = interrupt(payload)
    if not isinstance(decision, dict):
        decision = {"action": "continue", "feedback": ""}
    action = str(decision.get("action") or "continue").strip() or "continue"
    feedback = _normalize_human_feedback(decision.get("feedback"))
    result = {
        "plan_review_action": action,
        "plan_review_feedback": feedback if action == "replan_with_feedback" else "",
    }
    writer(
        evt.node_result(
            "review_plan",
            run_id=context_id,
            content=json.dumps(result, ensure_ascii=False, indent=2),
            data={"round": state.get("round", 0), **result},
        )
    )
    return result


async def dispatch_tasks(_: ResearchState) -> dict:
    return {}


# ---------------------------------------------------------------------------
# research_task  (one per Send, runs in parallel)
# ---------------------------------------------------------------------------

async def research_task(state: TaskState, config: RunnableConfig, writer: StreamWriter) -> dict:
    task = state["task"]
    task_id = task["id"]
    context_id = state["context_id"]
    factory = config["configurable"]["factory"]

    final_answer: str | None = None
    final_error: err.AgentError | None = None
    result_sent = False
    cancelled = False
    remote_task_id: str | None = None
    client = None

    trace.record("graph_task_start", run_id=context_id, task_id=task_id, content=task["mission"])
    try:
        client = factory.create(get_agent_card())
        mission = task["encoded_mission"]
        msg = new_text_message(mission, context_id=context_id)
        req = SendMessageRequest(message=msg)

        async for event in client.send_message(req):
            if event.HasField("artifact_update"):
                remote_task_id = event.artifact_update.task_id or remote_task_id
                final_answer = get_artifact_text(event.artifact_update.artifact)
                if final_answer:
                    trace.record(
                        "graph_task_result",
                        run_id=context_id,
                        task_id=task_id,
                        ok=True,
                        content=final_answer[:500],
                    )
                    writer(evt.task_result(task_id, final_answer, run_id=context_id))
                    result_sent = True
            elif event.HasField("status_update"):
                remote_task_id = event.status_update.task_id or remote_task_id
                status_state = event.status_update.status.state
                if status_state == A2ATaskState.TASK_STATE_WORKING:
                    text = get_message_text(event.status_update.status.message)
                    if text:
                        try:
                            writer(evt.status_text(task_id, text, run_id=context_id))
                        except ValueError as e:
                            status_error = err.exception_error(e, source=SOURCE, code="A2A_STATUS_FORMAT_ERROR")
                            writer(evt.observation(task_id, status_error.message, run_id=context_id))
                elif status_state in (A2ATaskState.TASK_STATE_FAILED, A2ATaskState.TASK_STATE_CANCELED):
                    text = get_message_text(event.status_update.status.message)
                    state_name = A2ATaskState.Name(status_state).removeprefix("TASK_STATE_").lower()
                    final_error = err.task_state_error(state_name, text or "task ended before producing an artifact")
                    final_answer = err.failed_text(final_error)
                    break
    except asyncio.CancelledError:
        cancelled = True
        if client is not None and remote_task_id:
            with contextlib.suppress(Exception):
                await client.cancel_task(CancelTaskRequest(id=remote_task_id))
        raise
    except Exception as e:
        final_error = err.exception_error(e, source=SOURCE, code="A2A_NETWORK_ERROR")
        final_answer = err.failed_text(final_error)
    finally:
        if cancelled:
            trace.record(
                "graph_task_done",
                run_id=context_id,
                task_id=task_id,
                ok=False,
                error_code="CLIENT_DISCONNECTED",
            )
        else:
            if not result_sent:
                if final_answer is None:
                    final_error = err.no_result()
                    final_answer = err.failed_text(final_error)
                trace.record(
                    "graph_task_result",
                    run_id=context_id,
                    task_id=task_id,
                    ok=final_error is None,
                    error_code=final_error.code if final_error else None,
                    content=final_answer[:500],
                )
                writer(evt.task_result(task_id, final_answer, run_id=context_id, error=final_error))
            trace.record("graph_task_done", run_id=context_id, task_id=task_id)
            writer(
                evt.node_result(
                    "research_task",
                    run_id=context_id,
                    task_id=task_id,
                    content=final_answer,
                    data={
                        "task": task,
                        "result": final_answer,
                        "ok": final_error is None,
                        "error": final_error.to_dict() if final_error else None,
                    },
                )
            )
            writer(evt.task_done(task_id, run_id=context_id))

    return {
        "subagent_results": {
            task_id: {
                "task_id": task_id,
                "task": task,
                "result": final_answer or err.failed_text(err.no_result()),
                "ok": final_error is None,
                "error": final_error.to_dict() if final_error else None,
            }
        }
    }


# ---------------------------------------------------------------------------
# collect_results  (fan-in: parse results, reflect on failures)
# ---------------------------------------------------------------------------

async def collect_results(state: ResearchState, writer: StreamWriter) -> dict:
    tasks = state.get("tasks", [])
    subagent_results = state.get("subagent_results", {})
    round_results = [
        subagent_results[task["id"]]
        for task in tasks
        if task.get("id") in subagent_results
    ]
    raw_results = [entry["result"] for entry in round_results]
    histories, citations = _parse_findings(raw_results)
    for entry, history in zip(round_results, histories, strict=False):
        task = entry.get("task") or {}
        run_context.publish_history(state["context_id"], task.get("id", ""), history)

    any_failed = any(not entry.get("ok", True) or err.is_failure_text(entry["result"]) for entry in round_results)
    if any_failed:
        await _reflect(state["query"], tasks, round_results)

    result = {
        "known_citations": citations,
        "round_histories": histories,
    }
    writer(
        evt.node_result(
            "collect_results",
            run_id=state["context_id"],
            content=json.dumps(citations, ensure_ascii=False, indent=2),
            data={
                "new_result_count": len(round_results),
                "history_count": len(histories),
                "citation_count": len(citations),
                "citations": _summarize_citations(citations),
                "history_previews": _summarize_histories(histories),
                "task_ids": [entry["task_id"] for entry in round_results],
                "any_failed": any_failed,
                "round": state.get("round", 0),
            },
        )
    )
    return result


# ---------------------------------------------------------------------------
# synthesize_answer
# ---------------------------------------------------------------------------

async def synthesize_answer(state: ResearchState, writer: StreamWriter) -> dict:
    context_id = state["context_id"]
    trace.record("synthesis_start", run_id=context_id, data={"round": state.get("round", 0)})
    writer(evt.synthesizing(run_id=context_id))
    logger.info("Synthesis node entered", extra={"run_id": context_id, "round": state.get("round", 0)})
    synthesis_inputs = list(state["round_histories"])
    prior_answer = None
    if state.get("current_answer"):
        prior_answer = strip_sources_section(state["current_answer"])
    run_citations = state.get("known_citations", [])
    evidence_evaluation: dict = {}
    draft_answer_preview = ""
    critical_review: dict = {}
    simple_evaluation: dict = {}
    try:
        synthesis = await synthesize_with_evaluation(
            state["rewritten_query"],
            synthesis_inputs,
            run_citations,
            prior_answer=prior_answer,
        )
        final = synthesis.answer
        evidence_evaluation = synthesis.evidence_evaluation
        draft_answer_preview = _preview_text(synthesis.draft_answer, 500)
        critical_review = synthesis.critical_review
        simple_evaluation = synthesis.simple_evaluation
        logger.info("Synthesis model call finished", extra={"run_id": context_id, "round": state.get("round", 0), "answer_length": len(final)})
    except Exception:
        logger.warning("Synthesis failed, falling back to concatenation", exc_info=True)
        final = "\n\n".join(
            f"Agent {i + 1} findings:\n{r}" for i, r in enumerate(state["round_histories"])
        )
    logger.info(
        "Synthesis completed",
        extra={
            "run_id": context_id,
            "round": state.get("round", 0),
            "answer_length": len(final),
        },
    )
    writer(
        evt.node_result(
            "synthesize_answer",
            run_id=context_id,
            content=_preview_text(final, 600),
            data={
                "query": state["rewritten_query"],
                "history_count": len(state["round_histories"]),
                "history_previews": _summarize_histories(state["round_histories"]),
                "prior_answer_length": len(prior_answer or ""),
                "citation_count": len(run_citations),
                "citations": _summarize_citations(run_citations),
                "evidence_evaluation": evidence_evaluation,
                "draft_answer_preview": draft_answer_preview,
                "critical_review": critical_review,
                "simple_evaluation": simple_evaluation,
                "current_answer_length": len(final),
                "current_answer_preview": _preview_text(final, 300),
                "round": state.get("round", 0),
            },
        )
    )
    logger.info("Synthesis node_result emitted", extra={"run_id": context_id, "round": state.get("round", 0)})
    return {
        "current_answer": final,
        "round_histories": [],
    }


# ---------------------------------------------------------------------------
# critique_answer
# ---------------------------------------------------------------------------

async def critique_answer(state: ResearchState, writer: StreamWriter) -> dict:
    context_id = state["context_id"]
    round_num = state.get("round", 0)
    prompt = CRITIC_PROMPT.format(
        query=state["rewritten_query"],
        answer=state["current_answer"],
        max_missions=CRITIC_MAX_MISSIONS,
    )
    try:
        response = await llm.chat([{"role": "user", "content": prompt}])
        data = json.loads(response)
        ok = bool(data.get("ok", True))
        gaps = [] if ok else [g for g in data.get("gaps", []) if g][:CRITIC_MAX_MISSIONS]
        planning_guidance = "" if ok else (data.get("planning_guidance", "") or "").strip()
    except Exception:
        logger.warning("Critic failed, treating answer as sufficient", exc_info=True)
        ok, gaps, planning_guidance = True, [], ""

    critique_result = {
        "round": round_num,
        "ok": ok,
        "gaps": gaps if not ok else [],
        "planning_guidance": planning_guidance if not ok else "",
    }
    writer(evt.critique(ok, gaps if not ok else [], run_id=context_id))
    writer(
        evt.node_result(
            "critique_answer",
            run_id=context_id,
            content=json.dumps(critique_result, ensure_ascii=False, indent=2),
            data=critique_result,
        )
    )
    trace.record("critique", run_id=context_id, data={"round": round_num, "ok": ok, "gaps": len(gaps)})

    return {
        "critique_ok": ok,
        "critique_gaps": gaps if not ok else [],
        "planning_guidance": planning_guidance if not ok else "",
        "round": round_num + 1,
    }


async def review_critique(state: ResearchState, writer: StreamWriter) -> dict:
    context_id = state["context_id"]
    if not state.get("human_in_loop", True):
        if state.get("critique_ok") or not state.get("planning_guidance"):
            return {
                "critique_review_action": "finish",
                "critique_review_feedback": "",
                "planning_guidance": "",
                "critique_ok": True,
            }
        if state.get("round", 0) >= MAX_CRITIC_ROUNDS:
            return {
                "critique_review_action": "finish",
                "critique_review_feedback": "",
                "planning_guidance": "",
                "critique_ok": True,
            }
        return {
            "critique_review_action": "continue",
            "critique_review_feedback": "",
            "planning_guidance": state.get("planning_guidance", "") or DEFAULT_CONTINUE_GUIDANCE,
            "critique_ok": False,
        }

    if not state.get("critique_ok") and state.get("round", 0) >= MAX_CRITIC_ROUNDS:
        message = "Reached the maximum follow-up rounds. The run will end with the current answer."
        writer(
            evt.human_review(
                "critique_limit",
                message,
                [],
                run_id=context_id,
                payload={
                    "stage": "critique_limit",
                    "run_id": context_id,
                    "round": state.get("round", 0),
                    "max_rounds": MAX_CRITIC_ROUNDS,
                    "gaps": state.get("critique_gaps", []),
                    "current_answer": state.get("current_answer", ""),
                },
            )
        )
        result = {
            "critique_review_action": "finish",
            "critique_review_feedback": "",
            "planning_guidance": "",
            "critique_ok": True,
        }
        writer(
            evt.node_result(
                "review_critique",
                run_id=context_id,
                content=json.dumps(result, ensure_ascii=False, indent=2),
                data={"round": state.get("round", 0), "reason": "max_rounds_reached", **result},
            )
        )
        return result

    payload = {
        "stage": "critique",
        "run_id": context_id,
        "round": state.get("round", 0),
        "max_rounds": MAX_CRITIC_ROUNDS,
        "ok": state.get("critique_ok", False),
        "gaps": state.get("critique_gaps", []),
        "planning_guidance": state.get("planning_guidance", ""),
        "current_answer": state.get("current_answer", ""),
        "current_answer_preview": _preview_text(state.get("current_answer", ""), 500),
        "current_answer_length": len(state.get("current_answer", "")),
        "prompt": "Review the synthesized answer and decide whether to finish or continue research.",
        "options": [
            {"action": "finish", "label": "Finish"},
            {"action": "continue", "label": "Continue"},
            {"action": "continue_with_feedback", "label": "Continue with Feedback"},
        ],
    }
    writer(
        evt.node_result(
            "review_critique",
            run_id=context_id,
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            data=payload,
            status="waiting",
        )
    )
    decision = interrupt(payload)
    if not isinstance(decision, dict):
        decision = {"action": "finish", "feedback": ""}
    action = str(decision.get("action") or "finish").strip() or "finish"
    feedback = _normalize_human_feedback(decision.get("feedback"))

    base_guidance = state.get("planning_guidance", "")
    if action == "finish":
        next_guidance = ""
        critique_ok = True
    else:
        if not base_guidance:
            base_guidance = await _generate_continue_guidance(state)
        if action == "continue_with_feedback":
            next_guidance = _merge_guidance(base_guidance, feedback)
        else:
            next_guidance = base_guidance
        critique_ok = False

    result = {
        "critique_review_action": action,
        "critique_review_feedback": feedback if action == "continue_with_feedback" else "",
        "planning_guidance": next_guidance,
        "critique_ok": critique_ok,
    }
    writer(
        evt.node_result(
            "review_critique",
            run_id=context_id,
            content=json.dumps(result, ensure_ascii=False, indent=2),
            data={"round": state.get("round", 0), **result},
        )
    )
    return result


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def route_after_plan_review(state: ResearchState) -> str:
    action = state.get("plan_review_action") or "continue"
    if action in {"replan", "replan_with_feedback"}:
        return "plan_tasks"
    if not state.get("tasks"):
        return "collect_results"
    return "dispatch_tasks"


def route_after_critique(state: ResearchState) -> str:
    if (state.get("critique_review_action") or "finish") == "finish":
        return "finalize"
    return "plan_tasks"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

async def _reflect(query: str, tasks: list[dict], subagent_results: list[dict]) -> None:
    outcomes = "\n".join(
        f"Subagent {i + 1} ({t['mission'][:80]}): {'failed' if (not r.get('ok', True) or err.is_failure_text(r['result'])) else 'succeeded'}"
        for i, (t, r) in enumerate(zip(tasks, subagent_results, strict=False))
    )
    prompt = ORCHESTRATOR_REFLECT_PROMPT.format(
        query=query,
        tasks="\n".join(f"- {t['mission']}" for t in tasks),
        outcomes=outcomes,
    )
    try:
        reflection = await llm.complete([{"role": "user", "content": prompt}])
        if reflection.strip():
            await lt.add("orchestrator", reflection.strip())
    except Exception:
        logger.warning("Orchestrator reflection failed", exc_info=True)
