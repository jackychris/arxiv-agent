# graph/state.py
import operator
from typing import Annotated, TypedDict


def _merge_subagent_results(
    left: dict[str, dict] | None, right: dict[str, dict] | None
) -> dict[str, dict]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_citations(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    merged: dict[str, dict] = {}
    next_ref_num = 1

    for citation in list(left or []) + list(right or []):
        key = citation.get("key")
        if not key:
            continue

        existing = merged.get(key)
        candidate = dict(existing or {})
        candidate.update(citation)
        candidate["key"] = key
        candidate["cited"] = bool((existing or {}).get("cited")) or bool(citation.get("cited"))

        existing_title = str((existing or {}).get("title") or "").strip()
        incoming_title = str(citation.get("title") or "").strip()
        candidate["title"] = existing_title if len(existing_title) >= len(incoming_title) else incoming_title

        if existing and existing.get("ref_num") is not None:
            candidate["ref_num"] = existing["ref_num"]
        elif citation.get("ref_num") is not None:
            candidate["ref_num"] = citation["ref_num"]
        else:
            candidate["ref_num"] = next_ref_num

        merged[key] = candidate
        next_ref_num = max(next_ref_num, int(candidate["ref_num"]) + 1)

    return sorted(
        merged.values(),
        key=lambda c: (c.get("ref_num") is None, c.get("ref_num", 0), c.get("key", "")),
    )


class ResearchState(TypedDict):
    query: str
    rewritten_query: str
    context_id: str
    human_in_loop: bool
    tasks: list[dict]
    subagent_results: Annotated[dict[str, dict], _merge_subagent_results]  # keyed by task_id across parallel tasks
    known_citations: Annotated[list[dict], _merge_citations]               # round-level shared citations across completed work
    round_histories: list[str]                              # current round histories only
    current_answer: str                                     # answer draft carried across rounds
    round: int
    critique_ok: bool
    critique_gaps: list[str]
    planning_guidance: str
    plan_review_action: str
    plan_review_feedback: str
    critique_review_action: str
    critique_review_feedback: str


class TaskState(TypedDict):
    task: dict
    context_id: str
