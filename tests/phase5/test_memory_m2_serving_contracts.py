"""M2.4 Context Packet and Retrieval Trace contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel.canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope
from cognitive_kernel.serving_contracts import (
    CONTEXT_PACKET_STATES,
    CONTEXT_SOURCE_KINDS,
    RETRIEVAL_FUSION_STRATEGIES,
    RETRIEVAL_STAGE_KINDS,
    ContextPacket,
    ContextSelection,
    RetrievalTrace,
    RetrievalTraceStep,
)

NOW = "2026-08-06T22:00:00Z"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="owner-primary",
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def envelope(record_id: str, record_type: str, content: str) -> MemoryUnitEnvelope:
    return MemoryUnitEnvelope.create(
        scope=scope(),
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id="owner-primary",
        host_or_cluster_id="owner-primary",
        authority_role="registered_projection",
        deployment_profile="reversible_prototype",
        created_at=NOW,
        valid_from=NOW,
        valid_to=None,
        transaction_time=NOW,
        logical_clock=0,
        causal_parents=(),
        source_records=(),
        generation=0,
        state="committed",
        data_classification="private",
        retention_class="transient_web_or_tool_cache",
        deletion_state="active",
        provenance_digest=DIGEST_A,
        content_digest=content,
        writer="bounded-serving-prototype",
        workflow_or_request_id="request-1",
        idempotency_namespace="bounded-serving",
        idempotency_key=record_id,
    )


def selection(rank: int = 1) -> ContextSelection:
    return ContextSelection.create(
        record_id=f"record-{rank}",
        record_version_id=f"version-{rank}",
        source_kind="claim",
        authority_namespace_id="owner-primary",
        rank=rank,
        fused_score=0.8,
        content_digest=DIGEST_B,
        reason_codes=("lexical_match",),
        stale=False,
        selected_from_generation=3,
    )


def step() -> RetrievalTraceStep:
    return RetrievalTraceStep.create(
        stage_id="trace-1-stage-1",
        stage_kind="lexical_retrieval",
        outcome="completed",
        started_at=NOW,
        completed_at=NOW,
        input_record_ids=("seed-1",),
        output_record_ids=("record-1",),
        excluded_record_ids=(),
        reason_codes=("lexical_match",),
        index_kind="lexical",
        index_generation=3,
        fallback_used=False,
        stale_index_observed=False,
        metrics_digest=DIGEST_C,
    )


def trace() -> RetrievalTrace:
    return RetrievalTrace.create(
        envelope=envelope("trace-1", "retrieval_trace", DIGEST_C),
        trace_id="trace-1",
        request_id="request-1",
        query_digest=DIGEST_A,
        profile_id="serving-profile",
        fusion_strategy="weighted_score",
        steps=(step(),),
        selected_record_ids=("record-1",),
        excluded_record_ids=("record-2",),
        started_at=NOW,
        completed_at=NOW,
        fallback_used=False,
        stale_index_observed=False,
        trace_content_digest=DIGEST_C,
    )


def packet() -> ContextPacket:
    return ContextPacket.create(
        envelope=envelope("packet-1", "context_packet", DIGEST_B),
        packet_id="packet-1",
        request_id="request-1",
        trace_id="trace-1",
        query_digest=DIGEST_A,
        profile_id="serving-profile",
        packet_state="assembled",
        mission_node_ids=("mission-1",),
        selections=(selection(),),
        excluded_record_ids=("record-2",),
        assembled_at=NOW,
        expires_at=None,
        item_budget=3,
        byte_budget=4096,
        hydrated_item_count=1,
        hydrated_byte_count=128,
        selection_generation=3,
        fallback_used=False,
        degraded=False,
        packet_content_digest=DIGEST_B,
    )


def test_registered_vocabularies_cover_m2_4_contracts() -> None:
    assert "assembled" in CONTEXT_PACKET_STATES
    assert "projection" in CONTEXT_SOURCE_KINDS
    assert "graph_expansion" in RETRIEVAL_STAGE_KINDS
    assert "weighted_score" in RETRIEVAL_FUSION_STRATEGIES


def test_context_selection_is_digest_bound() -> None:
    value = selection()
    value.validate()
    assert value.selection_sha256 == canonical_sha256(value.semantic_record())


def test_context_selection_rejects_unknown_source_kind() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(selection(), source_kind="unknown").validate()


def test_context_selection_rejects_duplicate_reason_codes() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            selection(),
            reason_codes=("lexical_match", "lexical_match"),
        ).validate()


def test_trace_step_requires_registered_stage_kind() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(step(), stage_kind="magic").validate()


def test_trace_step_generation_requires_index_kind() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(step(), index_kind=None).validate()


def test_retrieval_trace_is_scope_and_digest_bound() -> None:
    value = trace()
    value.validate()
    assert value.scope.storage_scope() == scope().storage_scope()
    assert value.trace_sha256 == canonical_sha256(value.semantic_record())


def test_retrieval_trace_rejects_selected_excluded_overlap() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            trace(),
            excluded_record_ids=("record-1",),
        ).validate()


def test_retrieval_trace_requires_correct_envelope_type() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            trace(),
            envelope=envelope("trace-1", "context_packet", DIGEST_C),
        ).validate()


def test_context_packet_is_bounded_by_selected_profile() -> None:
    value = packet()
    value.validate()
    assert value.hydrated_item_count <= value.item_budget
    assert value.hydrated_byte_count <= value.byte_budget


def test_context_packet_rejects_noncontiguous_ranks() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            packet(),
            selections=(selection(rank=2),),
        ).validate()


def test_context_packet_rejects_item_budget_overrun() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            packet(),
            item_budget=1,
            hydrated_item_count=2,
        ).validate()


def test_context_packet_rejects_byte_budget_overrun() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            packet(),
            byte_budget=64,
            hydrated_byte_count=128,
        ).validate()


def test_context_packet_rejects_digest_tampering() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(packet(), packet_sha256="0" * 64).validate()
