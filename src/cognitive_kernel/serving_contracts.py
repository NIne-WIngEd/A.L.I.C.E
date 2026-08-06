"""Memory M2.4 Context Packet and Retrieval Trace contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_confidence,
    require_identifier,
    require_sha256,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope

SERVING_CONTRACT_SCHEMA_VERSION = "1.0.0"

CONTEXT_PACKET_STATES = frozenset(
    {
        "assembled",
        "partial",
        "degraded",
        "stale_fallback",
        "superseded",
        "expired",
        "deleted",
    }
)

RETRIEVAL_STAGE_KINDS = frozenset(
    {
        "query_normalization",
        "lexical_retrieval",
        "vector_retrieval",
        "graph_expansion",
        "temporal_filtering",
        "mission_filtering",
        "fusion",
        "hydration",
        "fallback",
        "packet_assembly",
    }
)

RETRIEVAL_STAGE_OUTCOMES = frozenset(
    {
        "completed",
        "partial",
        "empty",
        "skipped",
        "stale",
        "unavailable",
        "fallback_used",
        "failed",
    }
)

CONTEXT_SOURCE_KINDS = frozenset(
    {
        "claim",
        "episode",
        "projection",
        "evidence",
        "mission",
        "conversation",
        "tool",
        "web",
        "file",
        "other",
    }
)

RETRIEVAL_FUSION_STRATEGIES = frozenset(
    {
        "weighted_score",
        "reciprocal_rank",
        "lexical_first",
        "vector_first",
        "graph_first",
        "custom",
    }
)


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CognitiveKernelContractError(
            f"{field} must be a positive integer"
        )
    return value


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _optional_identifier(
    value: object | None,
    field: str,
) -> str | None:
    return (
        require_identifier(value, field)
        if value is not None
        else None
    )


def _optional_timestamp(
    value: object | None,
    field: str,
) -> str | None:
    return (
        normalize_timestamp(value, field)
        if value is not None
        else None
    )


def _sorted_identifiers(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(normalize_identifier_sequence(values, field))
    )


@dataclass(frozen=True)
class ContextSelection:
    """One authority-aware record selected for a Context Packet."""

    record_id: str
    record_version_id: str
    source_kind: str
    authority_namespace_id: str
    rank: int
    fused_score: float
    content_digest: str
    reason_codes: tuple[str, ...]
    stale: bool
    selected_from_generation: int
    selection_sha256: str

    @classmethod
    def create(
        cls,
        *,
        record_id: object,
        record_version_id: object,
        source_kind: object,
        authority_namespace_id: object,
        rank: object,
        fused_score: object,
        content_digest: object,
        reason_codes: Iterable[object] = (),
        stale: object = False,
        selected_from_generation: object = 0,
    ) -> "ContextSelection":
        if not isinstance(stale, bool):
            raise CognitiveKernelContractError(
                "stale must be boolean"
            )
        score = require_confidence(
            fused_score,
            "fused_score",
        )
        if score is None:
            raise CognitiveKernelContractError(
                "fused_score is required"
            )
        draft = cls(
            record_id=require_identifier(
                record_id,
                "record_id",
            ),
            record_version_id=require_identifier(
                record_version_id,
                "record_version_id",
            ),
            source_kind=require_identifier(
                source_kind,
                "source_kind",
            ),
            authority_namespace_id=require_identifier(
                authority_namespace_id,
                "authority_namespace_id",
            ),
            rank=_positive_integer(rank, "rank"),
            fused_score=score,
            content_digest=require_sha256(
                content_digest,
                "content_digest",
            ),
            reason_codes=_sorted_identifiers(
                reason_codes,
                "reason_codes",
            ),
            stale=stale,
            selected_from_generation=_non_negative_integer(
                selected_from_generation,
                "selected_from_generation",
            ),
            selection_sha256="",
        )
        value = cls(
            **{
                **draft.__dict__,
                "selection_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": SERVING_CONTRACT_SCHEMA_VERSION,
            "record_id": self.record_id,
            "record_version_id": self.record_version_id,
            "source_kind": self.source_kind,
            "authority_namespace_id": self.authority_namespace_id,
            "rank": self.rank,
            "fused_score": self.fused_score,
            "content_digest": self.content_digest,
            "reason_codes": list(self.reason_codes),
            "stale": self.stale,
            "selected_from_generation": (
                self.selected_from_generation
            ),
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "selection_sha256": self.selection_sha256,
        }

    def validate(self) -> None:
        if self.source_kind not in CONTEXT_SOURCE_KINDS:
            raise CognitiveKernelContractError(
                "source_kind is not registered"
            )
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise CognitiveKernelContractError(
                "reason_codes must be unique and sorted"
            )
        if canonical_sha256(self.semantic_record()) != self.selection_sha256:
            raise CognitiveKernelContractError(
                "ContextSelection digest mismatch"
            )


@dataclass(frozen=True)
class RetrievalTraceStep:
    """One explainable stage in a bounded retrieval execution."""

    stage_id: str
    stage_kind: str
    outcome: str
    started_at: str
    completed_at: str
    input_record_ids: tuple[str, ...]
    output_record_ids: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    index_kind: str | None
    index_generation: int | None
    fallback_used: bool
    stale_index_observed: bool
    metrics_digest: str
    step_sha256: str

    @classmethod
    def create(
        cls,
        *,
        stage_id: object,
        stage_kind: object,
        outcome: object,
        started_at: object,
        completed_at: object,
        input_record_ids: Iterable[object] = (),
        output_record_ids: Iterable[object] = (),
        excluded_record_ids: Iterable[object] = (),
        reason_codes: Iterable[object] = (),
        index_kind: object | None = None,
        index_generation: object | None = None,
        fallback_used: object = False,
        stale_index_observed: object = False,
        metrics_digest: object,
    ) -> "RetrievalTraceStep":
        if not isinstance(fallback_used, bool):
            raise CognitiveKernelContractError(
                "fallback_used must be boolean"
            )
        if not isinstance(stale_index_observed, bool):
            raise CognitiveKernelContractError(
                "stale_index_observed must be boolean"
            )
        started = normalize_timestamp(started_at, "started_at")
        completed = normalize_timestamp(completed_at, "completed_at")
        if completed < started:
            raise CognitiveKernelContractError(
                "completed_at may not precede started_at"
            )
        generation = (
            _non_negative_integer(
                index_generation,
                "index_generation",
            )
            if index_generation is not None
            else None
        )
        draft = cls(
            stage_id=require_identifier(stage_id, "stage_id"),
            stage_kind=require_identifier(
                stage_kind,
                "stage_kind",
            ),
            outcome=require_identifier(outcome, "outcome"),
            started_at=started,
            completed_at=completed,
            input_record_ids=_sorted_identifiers(
                input_record_ids,
                "input_record_ids",
            ),
            output_record_ids=_sorted_identifiers(
                output_record_ids,
                "output_record_ids",
            ),
            excluded_record_ids=_sorted_identifiers(
                excluded_record_ids,
                "excluded_record_ids",
            ),
            reason_codes=_sorted_identifiers(
                reason_codes,
                "reason_codes",
            ),
            index_kind=_optional_identifier(
                index_kind,
                "index_kind",
            ),
            index_generation=generation,
            fallback_used=fallback_used,
            stale_index_observed=stale_index_observed,
            metrics_digest=require_sha256(
                metrics_digest,
                "metrics_digest",
            ),
            step_sha256="",
        )
        value = cls(
            **{
                **draft.__dict__,
                "step_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": SERVING_CONTRACT_SCHEMA_VERSION,
            "stage_id": self.stage_id,
            "stage_kind": self.stage_kind,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "input_record_ids": list(self.input_record_ids),
            "output_record_ids": list(self.output_record_ids),
            "excluded_record_ids": list(self.excluded_record_ids),
            "reason_codes": list(self.reason_codes),
            "index_kind": self.index_kind,
            "index_generation": self.index_generation,
            "fallback_used": self.fallback_used,
            "stale_index_observed": self.stale_index_observed,
            "metrics_digest": self.metrics_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "step_sha256": self.step_sha256,
        }

    def validate(self) -> None:
        if self.stage_kind not in RETRIEVAL_STAGE_KINDS:
            raise CognitiveKernelContractError(
                "stage_kind is not registered"
            )
        if self.outcome not in RETRIEVAL_STAGE_OUTCOMES:
            raise CognitiveKernelContractError(
                "outcome is not registered"
            )
        if self.index_generation is not None and self.index_kind is None:
            raise CognitiveKernelContractError(
                "index_generation requires index_kind"
            )
        for field, value in (
            ("input_record_ids", self.input_record_ids),
            ("output_record_ids", self.output_record_ids),
            ("excluded_record_ids", self.excluded_record_ids),
            ("reason_codes", self.reason_codes),
        ):
            if tuple(sorted(set(value))) != value:
                raise CognitiveKernelContractError(
                    f"{field} must be unique and sorted"
                )
        if canonical_sha256(self.semantic_record()) != self.step_sha256:
            raise CognitiveKernelContractError(
                "RetrievalTraceStep digest mismatch"
            )


@dataclass(frozen=True)
class RetrievalTrace:
    """Complete explainable trace for one bounded retrieval request."""

    envelope: MemoryUnitEnvelope
    trace_id: str
    request_id: str
    query_digest: str
    profile_id: str
    fusion_strategy: str
    steps: tuple[RetrievalTraceStep, ...]
    selected_record_ids: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]
    started_at: str
    completed_at: str
    fallback_used: bool
    stale_index_observed: bool
    trace_content_digest: str
    trace_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        trace_id: object,
        request_id: object,
        query_digest: object,
        profile_id: object,
        fusion_strategy: object,
        steps: Iterable[RetrievalTraceStep],
        selected_record_ids: Iterable[object] = (),
        excluded_record_ids: Iterable[object] = (),
        started_at: object,
        completed_at: object,
        fallback_used: object,
        stale_index_observed: object,
        trace_content_digest: object,
    ) -> "RetrievalTrace":
        if not isinstance(envelope, MemoryUnitEnvelope):
            raise CognitiveKernelContractError(
                "envelope must be a MemoryUnitEnvelope"
            )
        envelope.validate()
        if not isinstance(fallback_used, bool):
            raise CognitiveKernelContractError(
                "fallback_used must be boolean"
            )
        if not isinstance(stale_index_observed, bool):
            raise CognitiveKernelContractError(
                "stale_index_observed must be boolean"
            )
        normalized_steps = tuple(steps)
        if not normalized_steps:
            raise CognitiveKernelContractError(
                "steps may not be empty"
            )
        for step in normalized_steps:
            if not isinstance(step, RetrievalTraceStep):
                raise CognitiveKernelContractError(
                    "steps must contain RetrievalTraceStep values"
                )
            step.validate()
        started = normalize_timestamp(started_at, "started_at")
        completed = normalize_timestamp(completed_at, "completed_at")
        if completed < started:
            raise CognitiveKernelContractError(
                "completed_at may not precede started_at"
            )
        draft = cls(
            envelope=envelope,
            trace_id=require_identifier(trace_id, "trace_id"),
            request_id=require_identifier(request_id, "request_id"),
            query_digest=require_sha256(
                query_digest,
                "query_digest",
            ),
            profile_id=require_identifier(profile_id, "profile_id"),
            fusion_strategy=require_identifier(
                fusion_strategy,
                "fusion_strategy",
            ),
            steps=normalized_steps,
            selected_record_ids=_sorted_identifiers(
                selected_record_ids,
                "selected_record_ids",
            ),
            excluded_record_ids=_sorted_identifiers(
                excluded_record_ids,
                "excluded_record_ids",
            ),
            started_at=started,
            completed_at=completed,
            fallback_used=fallback_used,
            stale_index_observed=stale_index_observed,
            trace_content_digest=require_sha256(
                trace_content_digest,
                "trace_content_digest",
            ),
            trace_sha256="",
        )
        value = cls(
            **{
                **draft.__dict__,
                "trace_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": SERVING_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "query_digest": self.query_digest,
            "profile_id": self.profile_id,
            "fusion_strategy": self.fusion_strategy,
            "steps": [step.metadata_record() for step in self.steps],
            "selected_record_ids": list(self.selected_record_ids),
            "excluded_record_ids": list(self.excluded_record_ids),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "fallback_used": self.fallback_used,
            "stale_index_observed": self.stale_index_observed,
            "trace_content_digest": self.trace_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "trace_sha256": self.trace_sha256,
        }

    def validate(self) -> None:
        if self.envelope.record_id != self.trace_id:
            raise CognitiveKernelContractError(
                "envelope record_id must equal trace_id"
            )
        if self.envelope.record_type != "retrieval_trace":
            raise CognitiveKernelContractError(
                "envelope record_type must be retrieval_trace"
            )
        if self.fusion_strategy not in RETRIEVAL_FUSION_STRATEGIES:
            raise CognitiveKernelContractError(
                "fusion_strategy is not registered"
            )
        if tuple(sorted(set(self.selected_record_ids))) != (
            self.selected_record_ids
        ):
            raise CognitiveKernelContractError(
                "selected_record_ids must be unique and sorted"
            )
        if tuple(sorted(set(self.excluded_record_ids))) != (
            self.excluded_record_ids
        ):
            raise CognitiveKernelContractError(
                "excluded_record_ids must be unique and sorted"
            )
        if set(self.selected_record_ids) & set(self.excluded_record_ids):
            raise CognitiveKernelContractError(
                "selected and excluded records may not overlap"
            )
        if canonical_sha256(self.semantic_record()) != self.trace_sha256:
            raise CognitiveKernelContractError(
                "RetrievalTrace digest mismatch"
            )


@dataclass(frozen=True)
class ContextPacket:
    """Bounded context assembly contract around runtime-held full content."""

    envelope: MemoryUnitEnvelope
    packet_id: str
    request_id: str
    trace_id: str
    query_digest: str
    profile_id: str
    packet_state: str
    mission_node_ids: tuple[str, ...]
    selections: tuple[ContextSelection, ...]
    excluded_record_ids: tuple[str, ...]
    assembled_at: str
    expires_at: str | None
    item_budget: int
    byte_budget: int
    hydrated_item_count: int
    hydrated_byte_count: int
    selection_generation: int
    fallback_used: bool
    degraded: bool
    packet_content_digest: str
    packet_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        packet_id: object,
        request_id: object,
        trace_id: object,
        query_digest: object,
        profile_id: object,
        packet_state: object,
        mission_node_ids: Iterable[object] = (),
        selections: Iterable[ContextSelection] = (),
        excluded_record_ids: Iterable[object] = (),
        assembled_at: object,
        expires_at: object | None,
        item_budget: object,
        byte_budget: object,
        hydrated_item_count: object,
        hydrated_byte_count: object,
        selection_generation: object,
        fallback_used: object,
        degraded: object,
        packet_content_digest: object,
    ) -> "ContextPacket":
        if not isinstance(envelope, MemoryUnitEnvelope):
            raise CognitiveKernelContractError(
                "envelope must be a MemoryUnitEnvelope"
            )
        envelope.validate()
        if not isinstance(fallback_used, bool):
            raise CognitiveKernelContractError(
                "fallback_used must be boolean"
            )
        if not isinstance(degraded, bool):
            raise CognitiveKernelContractError(
                "degraded must be boolean"
            )
        normalized_selections = tuple(selections)
        for selection in normalized_selections:
            if not isinstance(selection, ContextSelection):
                raise CognitiveKernelContractError(
                    "selections must contain ContextSelection values"
                )
            selection.validate()
        assembled = normalize_timestamp(assembled_at, "assembled_at")
        expires = _optional_timestamp(expires_at, "expires_at")
        if expires is not None and expires < assembled:
            raise CognitiveKernelContractError(
                "expires_at may not precede assembled_at"
            )
        draft = cls(
            envelope=envelope,
            packet_id=require_identifier(packet_id, "packet_id"),
            request_id=require_identifier(request_id, "request_id"),
            trace_id=require_identifier(trace_id, "trace_id"),
            query_digest=require_sha256(
                query_digest,
                "query_digest",
            ),
            profile_id=require_identifier(profile_id, "profile_id"),
            packet_state=require_identifier(
                packet_state,
                "packet_state",
            ),
            mission_node_ids=_sorted_identifiers(
                mission_node_ids,
                "mission_node_ids",
            ),
            selections=normalized_selections,
            excluded_record_ids=_sorted_identifiers(
                excluded_record_ids,
                "excluded_record_ids",
            ),
            assembled_at=assembled,
            expires_at=expires,
            item_budget=_positive_integer(
                item_budget,
                "item_budget",
            ),
            byte_budget=_positive_integer(
                byte_budget,
                "byte_budget",
            ),
            hydrated_item_count=_non_negative_integer(
                hydrated_item_count,
                "hydrated_item_count",
            ),
            hydrated_byte_count=_non_negative_integer(
                hydrated_byte_count,
                "hydrated_byte_count",
            ),
            selection_generation=_non_negative_integer(
                selection_generation,
                "selection_generation",
            ),
            fallback_used=fallback_used,
            degraded=degraded,
            packet_content_digest=require_sha256(
                packet_content_digest,
                "packet_content_digest",
            ),
            packet_sha256="",
        )
        value = cls(
            **{
                **draft.__dict__,
                "packet_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": SERVING_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "packet_id": self.packet_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "query_digest": self.query_digest,
            "profile_id": self.profile_id,
            "packet_state": self.packet_state,
            "mission_node_ids": list(self.mission_node_ids),
            "selections": [
                selection.metadata_record()
                for selection in self.selections
            ],
            "excluded_record_ids": list(self.excluded_record_ids),
            "assembled_at": self.assembled_at,
            "expires_at": self.expires_at,
            "item_budget": self.item_budget,
            "byte_budget": self.byte_budget,
            "hydrated_item_count": self.hydrated_item_count,
            "hydrated_byte_count": self.hydrated_byte_count,
            "selection_generation": self.selection_generation,
            "fallback_used": self.fallback_used,
            "degraded": self.degraded,
            "packet_content_digest": self.packet_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "packet_sha256": self.packet_sha256,
        }

    def validate(self) -> None:
        if self.envelope.record_id != self.packet_id:
            raise CognitiveKernelContractError(
                "envelope record_id must equal packet_id"
            )
        if self.envelope.record_type != "context_packet":
            raise CognitiveKernelContractError(
                "envelope record_type must be context_packet"
            )
        if self.packet_state not in CONTEXT_PACKET_STATES:
            raise CognitiveKernelContractError(
                "packet_state is not registered"
            )
        ranks = [selection.rank for selection in self.selections]
        if ranks != list(range(1, len(ranks) + 1)):
            raise CognitiveKernelContractError(
                "selection ranks must be contiguous from one"
            )
        selected_ids = [
            selection.record_id
            for selection in self.selections
        ]
        if len(selected_ids) != len(set(selected_ids)):
            raise CognitiveKernelContractError(
                "selections may not repeat record_id"
            )
        if set(selected_ids) & set(self.excluded_record_ids):
            raise CognitiveKernelContractError(
                "selected and excluded records may not overlap"
            )
        if self.hydrated_item_count != len(self.selections):
            raise CognitiveKernelContractError(
                "hydrated_item_count must equal selection count"
            )
        if self.hydrated_item_count > self.item_budget:
            raise CognitiveKernelContractError(
                "hydrated item count exceeds the selected profile budget"
            )
        if self.hydrated_byte_count > self.byte_budget:
            raise CognitiveKernelContractError(
                "hydrated byte count exceeds the selected profile budget"
            )
        if canonical_sha256(self.semantic_record()) != self.packet_sha256:
            raise CognitiveKernelContractError(
                "ContextPacket digest mismatch"
            )
