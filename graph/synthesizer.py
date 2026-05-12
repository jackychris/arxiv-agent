import asyncio
import logging
import re

import llm
from config import SYNTHESIZER_MAP_OUTPUT_LIMIT, SYNTHESIZER_MAP_THRESHOLD
from prompts import (
    MAP_SUMMARIZE_PROMPT,
    SYNTHESIZE_INITIAL_PROMPT,
    SYNTHESIZE_REFINE_PROMPT,
)

logger = logging.getLogger(__name__)

_TITLE_MAX = 60
_SOURCES_HEADING_RE = re.compile(r"(?ims)^\s{0,3}#{1,6}\s*(sources used|sources|references)\s*$")
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?])")


def _format_citations(citations: list[dict], *, cited_only: bool = False) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations:
        if cited_only and not c.get("cited"):
            continue
        ref_num = c.get("ref_num")
        label = ref_num if ref_num is not None else "?"
        title = (c.get("title") or "").strip()
        if title:
            lines.append(f"[{label}] {c['key']} — {title[:_TITLE_MAX]}")
        else:
            lines.append(f"[{label}] {c['key']}")
    return "\n".join(lines)


def _append_sources(answer: str, citation_block: str) -> str:
    if not citation_block:
        return answer
    return f"{answer.rstrip()}\n\n## Sources Used\n{citation_block}"


def strip_sources_section(answer: str) -> str:
    match = _SOURCES_HEADING_RE.search(answer)
    if not match:
        return answer
    return answer[: match.start()].rstrip()


def _mark_cited_citations(answer: str, citations: list[dict]) -> list[dict]:
    cited_numbers = {int(match.group(1)) for match in _INLINE_CITATION_RE.finditer(answer)}
    for citation in citations:
        citation["cited"] = citation.get("ref_num") in cited_numbers
    return citations


def _sanitize_inline_citations(answer: str, citations: list[dict]) -> str:
    valid_numbers = {
        int(ref_num)
        for ref_num in (c.get("ref_num") for c in citations)
        if isinstance(ref_num, int)
    }

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid_numbers else ""

    sanitized = _INLINE_CITATION_RE.sub(replace, answer)
    sanitized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


async def _map_finding(finding: str, limit: int) -> str:
    if len(finding) <= limit:
        return finding
    prompt = MAP_SUMMARIZE_PROMPT.format(
        findings=finding,
        max_chars=limit,
    )
    try:
        return await llm.complete([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.warning("Map summarization failed, truncating: %s", e)
        return finding[:limit] + "\n[truncated]"


async def synthesize(
    query: str,
    findings: list[str],
    citations: list[dict],
    *,
    prior_answer: str | None = None,
) -> str:
    citation_block = _format_citations(citations)
    total = sum(len(f) for f in findings) + len(citation_block) + len(prior_answer or "")
    if total > SYNTHESIZER_MAP_THRESHOLD:
        findings_budget = max(SYNTHESIZER_MAP_THRESHOLD - len(citation_block), len(findings) * 500)
        per_finding_limit = min(findings_budget // max(len(findings), 1), SYNTHESIZER_MAP_OUTPUT_LIMIT)
        logger.info(
            "Input too large (%d chars), running map phase over %d findings (limit %d each)",
            total, len(findings), per_finding_limit,
        )
        findings = list(await asyncio.gather(*[_map_finding(f, per_finding_limit) for f in findings]))

    results_text = "\n\n".join(
        f"Agent {i + 1} findings:\n{r}" for i, r in enumerate(findings)
    )
    if prior_answer:
        prompt = SYNTHESIZE_REFINE_PROMPT.format(
            query=query,
            prior_answer=prior_answer,
            results=results_text,
            citations=citation_block,
        )
    else:
        prompt = SYNTHESIZE_INITIAL_PROMPT.format(
            query=query,
            results=results_text,
            citations=citation_block,
        )
    answer = await llm.complete([{"role": "user", "content": prompt}])
    clean_answer = _sanitize_inline_citations(strip_sources_section(answer), citations)
    updated_citations = _mark_cited_citations(clean_answer, citations)
    return _append_sources(clean_answer, _format_citations(updated_citations, cited_only=True))
