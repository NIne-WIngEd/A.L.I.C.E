"""Deterministic exact-extractive claim planning for P4.10b live sources."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .contracts import InformationQuery
from .freshness import InformationTemporallyQualifiedSource
from .grounding import InformationClaimDraft, InformationSupportSpan
from .grounding_policy import InformationGroundingPolicy

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+")


class InformationLiveClaimPlanningError(ValueError):
    """Raised when qualified source text cannot produce exact support spans."""


def _query_terms(query: InformationQuery) -> frozenset[str]:
    return frozenset(word.casefold() for word in _WORD.findall(query.text))


def _sentence_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            ranges.append((start, end))
        start = match.end()
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        ranges.append((start, end))
    return tuple(ranges)


def _candidate_score(
    text: str,
    *,
    query_terms: frozenset[str],
    start: int,
) -> tuple[int, int, int]:
    terms = frozenset(word.casefold() for word in _WORD.findall(text))
    overlap = len(query_terms & terms)
    sentence_signal = 1 if text[-1:] in {".", "!", "?"} else 0
    return (overlap, sentence_signal, min(len(text), 500) - start // 100_000)


@dataclass(frozen=True)
class DeterministicLiveExtractiveClaimPlanner:
    """Select at most two exact sentences without paraphrase or inference."""

    grounding_policy: InformationGroundingPolicy
    minimum_claim_characters: int = 40

    def __post_init__(self) -> None:
        self.grounding_policy.validate()
        if not 20 <= self.minimum_claim_characters <= 500:
            raise InformationLiveClaimPlanningError(
                "minimum_claim_characters is outside the bounded profile."
            )

    def plan(
        self,
        *,
        query: InformationQuery,
        qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
        maximum_sources: int,
    ) -> tuple[
        tuple[InformationTemporallyQualifiedSource, ...],
        tuple[InformationClaimDraft, ...],
    ]:
        query.validate()
        if (
            not isinstance(maximum_sources, int)
            or isinstance(maximum_sources, bool)
            or not 1 <= maximum_sources <= 2
        ):
            raise InformationLiveClaimPlanningError(
                "P4.10b grounded-source budget must be one or two."
            )
        query_terms = _query_terms(query)
        planned: list[
            tuple[
                tuple[int, int, int],
                int,
                InformationTemporallyQualifiedSource,
                int,
                int,
            ]
        ] = []
        seen_source_ids: set[str] = set()
        for source_index, qualified in enumerate(qualified_sources):
            source = qualified.inspected_source.source
            source.validate()
            if source.source_id in seen_source_ids:
                raise InformationLiveClaimPlanningError(
                    "Qualified live source identities must be unique."
                )
            seen_source_ids.add(source.source_id)
            best: tuple[tuple[int, int, int], int, int] | None = None
            for start, end in _sentence_ranges(source.normalized_text):
                length = end - start
                if (
                    length < self.minimum_claim_characters
                    or length > self.grounding_policy.max_support_span_characters
                ):
                    continue
                text = source.normalized_text[start:end]
                if "\n" in text or not any(character.isalpha() for character in text):
                    continue
                score = _candidate_score(text, query_terms=query_terms, start=start)
                candidate = (score, start, end)
                if best is None or candidate > best:
                    best = candidate
            if best is None:
                continue
            score, start, end = best
            planned.append((score, -source_index, qualified, start, end))
        planned.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = planned[:maximum_sources]
        selected.sort(key=lambda item: -item[1])
        sources: list[InformationTemporallyQualifiedSource] = []
        drafts: list[InformationClaimDraft] = []
        for ordinal, (_score, _source_order, qualified, start, end) in enumerate(
            selected,
            1,
        ):
            source = qualified.inspected_source.source
            text = source.normalized_text[start:end]
            span = InformationSupportSpan.create(
                source=source,
                start_character=start,
                end_character=end,
            )
            claim_id = "live-claim-" + hashlib.sha256(
                (
                    f"{query.content_sha256}\n{source.source_id}\n"
                    f"{start}\n{end}\n{span.support_sha256}"
                ).encode("utf-8")
            ).hexdigest()[:20]
            draft = InformationClaimDraft.create(
                claim_id=claim_id,
                text=text,
                knowledge_status="external_claim",
                confidence=0.80,
                support_spans=(span,),
            )
            sources.append(qualified)
            drafts.append(draft)
        return tuple(sources), tuple(drafts)
