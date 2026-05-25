from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_UNCERTAINTY_RE = re.compile(
    r"\b(partial|partly|limited|uncertain|unclear|insufficient|not enough|"
    r"mixed evidence|caveat|cannot determine|not conclusive)\b",
    re.IGNORECASE,
)
_INTERNAL_PROCESS_RE = re.compile(
    r"\b(evidence evaluator|critical review|draft answer|final assembler|"
    r"as an ai|i cannot browse|the system|tool call|observation:)\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(may|might|could|likely|appears|suggests|limited|unclear|uncertain|"
    r"not enough|insufficient|partial|caveat)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvaluationCriterion:
    key: str
    name: str
    weight: float
    score: float
    passed: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "weight": self.weight,
            "score": self.score,
            "passed": self.passed,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AnswerEvaluation:
    score: float
    passed: bool
    threshold: float
    criteria: list[EvaluationCriterion] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "passed": self.passed,
            "threshold": self.threshold,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "flags": self.flags,
            "recommendations": self.recommendations,
        }


class SimpleAnswerEvaluator:
    """Deterministic, lightweight quality gate for synthesized research answers."""

    def __init__(self, *, pass_threshold: float = 0.72):
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        *,
        query: str,
        answer: str,
        citations: list[dict],
        evidence_evaluation: dict | None = None,
        critical_review: dict | None = None,
    ) -> AnswerEvaluation:
        evidence_evaluation = evidence_evaluation or {}
        critical_review = critical_review or {}
        criteria = [
            self._criterion_citation_grounding(answer, citations),
            self._criterion_citation_coverage(answer, citations),
            self._criterion_query_focus(query, answer),
            self._criterion_answer_substance(answer),
            self._criterion_internal_leakage(answer),
            self._criterion_overconfidence(answer),
        ]
        weighted = sum(criterion.score * criterion.weight for criterion in criteria)
        total_weight = sum(criterion.weight for criterion in criteria) or 1.0
        score = weighted / total_weight
        flags = self._flags(criteria, evidence_evaluation, critical_review)
        recommendations = self._recommendations(criteria)
        return AnswerEvaluation(
            score=score,
            passed=score >= self.pass_threshold and not self._has_blocking_flag(flags),
            threshold=self.pass_threshold,
            criteria=criteria,
            flags=flags,
            recommendations=recommendations,
        )

    def _criterion_citation_grounding(self, answer: str, citations: list[dict]) -> EvaluationCriterion:
        if not citations:
            return EvaluationCriterion(
                "citation_grounding",
                "Citation grounding",
                0.26,
                1.0,
                True,
                "No citations were available for this answer.",
            )

        valid_refs = {
            int(ref_num)
            for ref_num in (citation.get("ref_num") for citation in citations)
            if isinstance(ref_num, int)
        }
        used_refs = {int(match.group(1)) for match in _INLINE_CITATION_RE.finditer(answer)}
        invalid_refs = sorted(used_refs - valid_refs)
        cited_available = sorted(used_refs & valid_refs)
        if invalid_refs:
            score = 0.0
            notes = f"Invalid citation refs found: {invalid_refs}."
        elif cited_available:
            score = min(1.0, 0.55 + 0.15 * len(cited_available))
            notes = f"Used {len(cited_available)} valid inline citation(s)."
        else:
            score = 0.15
            notes = "Available sources exist, but the answer has no valid inline citations."
        return EvaluationCriterion(
            "citation_grounding",
            "Citation grounding",
            0.26,
            score,
            score >= 0.7,
            notes,
        )

    def _criterion_citation_coverage(self, answer: str, citations: list[dict]) -> EvaluationCriterion:
        if not citations:
            score = 1.0
            notes = "No available citations to cover."
        else:
            used_refs = {int(match.group(1)) for match in _INLINE_CITATION_RE.finditer(answer)}
            valid_refs = {
                int(ref_num)
                for ref_num in (citation.get("ref_num") for citation in citations)
                if isinstance(ref_num, int)
            }
            coverage = len(used_refs & valid_refs) / max(len(valid_refs), 1)
            score = min(1.0, 0.45 + coverage)
            notes = f"Used {len(used_refs & valid_refs)} of {len(valid_refs)} available source(s)."
        return EvaluationCriterion(
            "citation_coverage",
            "Citation coverage",
            0.16,
            score,
            score >= 0.65,
            notes,
        )

    def _criterion_query_focus(self, query: str, answer: str) -> EvaluationCriterion:
        query_terms = _terms(query)
        if not query_terms:
            score = 0.75
            notes = "Query has no useful lexical terms for focus scoring."
        else:
            answer_terms = _terms(answer)
            overlap = len(query_terms & answer_terms) / len(query_terms)
            score = min(1.0, 0.35 + overlap)
            notes = f"Matched {len(query_terms & answer_terms)} of {len(query_terms)} query term(s)."
        return EvaluationCriterion(
            "query_focus",
            "Query focus",
            0.18,
            score,
            score >= 0.7,
            notes,
        )

    def _criterion_answer_substance(self, answer: str) -> EvaluationCriterion:
        body = _strip_sources(answer)
        word_count = len(re.findall(r"\w+", body))
        paragraph_count = len([part for part in re.split(r"\n\s*\n", body) if part.strip()])
        if word_count >= 120 and paragraph_count >= 2:
            score = 1.0
            notes = f"Answer has {word_count} words across {paragraph_count} paragraph(s)."
        elif word_count >= 60:
            score = 0.72
            notes = f"Answer is brief but substantive enough ({word_count} words)."
        else:
            score = 0.35
            notes = f"Answer appears too thin ({word_count} words)."
        return EvaluationCriterion(
            "answer_substance",
            "Answer substance",
            0.16,
            score,
            score >= 0.7,
            notes,
        )

    def _criterion_internal_leakage(self, answer: str) -> EvaluationCriterion:
        leaks = sorted({match.group(0).lower() for match in _INTERNAL_PROCESS_RE.finditer(answer)})
        score = 0.2 if leaks else 1.0
        notes = f"Internal process terms leaked: {leaks}." if leaks else "No internal process leakage found."
        return EvaluationCriterion(
            "internal_leakage",
            "Internal process leakage",
            0.14,
            score,
            score >= 0.7,
            notes,
        )

    def _criterion_overconfidence(self, answer: str) -> EvaluationCriterion:
        body = _strip_sources(answer)
        has_absolute = bool(
            re.search(r"\b(always|never|proves|guarantees|best|state of the art|sota)\b", body, re.IGNORECASE)
        )
        has_hedge = bool(_HEDGE_RE.search(body))
        if has_absolute and not has_hedge:
            score = 0.45
            notes = "Answer uses absolute claims without qualifying language."
        else:
            score = 1.0
            notes = "No obvious unqualified absolute claim found."
        return EvaluationCriterion(
            "overconfidence",
            "Overconfidence check",
            0.10,
            score,
            score >= 0.7,
            notes,
        )

    def _flags(
        self,
        criteria: list[EvaluationCriterion],
        evidence_evaluation: dict,
        critical_review: dict,
    ) -> list[str]:
        flags = [criterion.key for criterion in criteria if not criterion.passed]
        return flags

    def _recommendations(
        self, criteria: list[EvaluationCriterion]
    ) -> list[str]:
        guidance = {
            "citation_grounding": "Add or repair inline citations using only available source numbers.",
            "citation_coverage": "Use a broader subset of available sources when they support distinct claims.",
            "query_focus": "Tighten the answer around the user's query terms and requested scope.",
            "answer_substance": "Add enough concrete content to answer the query directly.",
            "internal_leakage": "Remove internal pipeline/process language from the final answer.",
            "overconfidence": "Qualify absolute claims unless the cited evidence directly supports them.",
        }
        return [guidance[c.key] for c in criteria if not c.passed and c.key in guidance]

    def _has_blocking_flag(self, flags: list[str]) -> bool:
        return any(
            flag in {"citation_grounding", "internal_leakage"}
            for flag in flags
        )


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        if token
        not in {
            "and",
            "are",
            "for",
            "how",
            "the",
            "this",
            "that",
            "what",
            "when",
            "where",
            "which",
            "with",
        }
    }


def _list_field(data: dict, key: str) -> list[Any]:
    value = data.get(key)
    return value if isinstance(value, list) else []


def _strip_sources(answer: str) -> str:
    match = re.search(r"(?im)^\s{0,3}#{1,6}\s*Sources Used\s*$", answer)
    return answer[: match.start()].strip() if match else answer.strip()
