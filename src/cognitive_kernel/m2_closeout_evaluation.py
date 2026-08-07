"""M2-CLOSEOUT synthetic cross-prototype integration evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Mapping

from .adjudication_contracts import (
    ClaimCandidate,
    ClaimEvidenceRelation,
)
from .bounded_serving_prototype import (
    BoundedServingProfile,
    open_bounded_serving_prototype,
)
from .canonical import canonical_sha256
from .claim_authority_prototype import (
    ClaimAuthorityAppendRequest,
    open_claim_authority_prototype,
)
from .claim_contracts import (
    CanonicalTaggedValue,
    ClaimIdentity,
    ClaimQualifier,
)
from .contracts import ProductHostScope
from .curation_contracts import CurationReceipt
from .deletion_prototype import (
    DeletionPropagationProfile,
    open_deletion_propagation_prototype,
)
from .durable_workflow_prototype import (
    DurableWorkflowProfile,
    open_durable_workflow_prototype,
)
from .m2_closeout import (
    M2CloseoutReport,
    M2ComponentEvaluation,
    ShadowMigrationAdmissionDecision,
)
from .memory_contracts import MemoryUnitEnvelope
from .projection_contracts import (
    EpisodeRecord,
    ProjectionVersion,
)
from .projection_prototype import (
    ProjectionGraphEdge,
    ProjectionPrototypeProfile,
    open_projection_prototype,
)
from .shadow_adjudication_prototype import (
    ShadowAdjudicationSubmission,
    open_shadow_adjudication_prototype,
)

EVALUATED_AT = "2026-08-06T23:30:00Z"
LATER = "2026-08-06T23:31:00Z"
FINAL = "2026-08-06T23:32:00Z"
PROVENANCE_DIGEST = "a" * 64
CONTENT_DIGEST = "b" * 64
REQUEST_DIGEST = "c" * 64
PROJECTION_REQUEST_DIGEST = "d" * 64

ADMITTED_STAGES = (
    "stage_a_inventory_registration",
    "stage_b_contract_adapters",
    "stage_c_destination_candidates",
    "stage_e_shadow_reads",
)
EXCLUDED_STAGES = (
    "stage_d_historical_backfill",
    "stage_f_controlled_write_mirroring",
    "stage_g_graph_vector_workflow_build",
    "stage_h_canary_authority",
    "stage_i_cutover",
    "stage_j_compatibility_operation",
)


def _scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="m2-closeout-host",
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def _namespace() -> str:
    return "owner-memory"


def _envelope(
    *,
    record_id: str,
    record_type: str,
    authority_role: str,
    source_records: tuple[str, ...] = (),
    generation: int = 0,
    writer: str = "m2-closeout-evaluator",
    workflow_id: str = "m2-closeout-evaluation",
) -> MemoryUnitEnvelope:
    return MemoryUnitEnvelope.create(
        scope=_scope(),
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id=_namespace(),
        host_or_cluster_id="m2-closeout-host",
        authority_role=authority_role,
        deployment_profile="single_workstation",
        created_at=EVALUATED_AT,
        valid_from=EVALUATED_AT,
        valid_to=None,
        transaction_time=EVALUATED_AT,
        logical_clock=generation,
        causal_parents=(),
        source_records=source_records,
        generation=generation,
        state="committed",
        data_classification="highly_sensitive",
        retention_class="high_value_experience",
        deletion_state="active",
        provenance_digest=PROVENANCE_DIGEST,
        content_digest=CONTENT_DIGEST,
        writer=writer,
        workflow_or_request_id=workflow_id,
        idempotency_namespace="m2_closeout",
        idempotency_key=record_id,
    )


def _text(value: str) -> CanonicalTaggedValue:
    return CanonicalTaggedValue.create(
        type_tag="text",
        value=value,
    )


def _qualifier(key: str, value: str) -> ClaimQualifier:
    return ClaimQualifier.create(
        key=key,
        value=_text(value),
    )


def _claim_identity() -> ClaimIdentity:
    return ClaimIdentity.create(
        envelope=_envelope(
            record_id="claim-1",
            record_type="claim_identity",
            authority_role="claim_authority",
        ),
        claim_id="claim-1",
        canonical_subject=CanonicalTaggedValue.create(
            type_tag="identifier",
            value="owner-primary",
        ),
        canonical_predicate="prefers_interface",
        canonical_value=_text("dark mode"),
        qualifiers=(),
        semantic_scope=(
            "user_interface",
            "owner_preference",
        ),
        canonicalization_version="1.0.0",
    )


def _claim_request() -> ClaimAuthorityAppendRequest:
    identity = _claim_identity()
    return ClaimAuthorityAppendRequest.create(
        identity=identity,
        version_envelope=_envelope(
            record_id="claim-version-1",
            record_type="claim_version",
            authority_role="claim_authority",
            source_records=("claim-1", "evidence-1"),
            workflow_id="claim-write-1",
        ),
        projection_envelope=_envelope(
            record_id="current-claim-1",
            record_type="current_claim_projection",
            authority_role="registered_projection",
            source_records=("claim-1", "claim-version-1"),
            workflow_id="claim-projection-1",
        ),
        value=_text("dark mode"),
        qualifiers=(_qualifier("context", "desktop"),),
        authority_class="owner_attested",
        confidence=0.99,
        adjudication_state="accepted",
        evidence_relation_ids=("relation-1",),
        conflict_set_id=None,
        correction_of=(),
        request_digest=REQUEST_DIGEST,
        expected_current_claim_version_id=None,
        validity_state="current",
        conflict_state="none",
        deletion_state="active",
    )


def _evidence_relation() -> ClaimEvidenceRelation:
    return ClaimEvidenceRelation.create(
        envelope=_envelope(
            record_id="relation-1",
            record_type="claim_evidence_relation",
            authority_role="candidate",
            source_records=("evidence-1", "candidate-1"),
        ),
        relation_id="relation-1",
        evidence_record_id="evidence-1",
        target_record_id="candidate-1",
        target_record_type="claim_candidate",
        relation_type="support",
        source_class="experience_ledger_event",
        source_authority_class="owner_attested",
        extractor_component_id="phase2-read-adapter-challenger",
        extractor_version="1.0.0",
        confidence=0.9,
    )


def _shadow_submission() -> ShadowAdjudicationSubmission:
    relation = _evidence_relation()
    candidate = ClaimCandidate.create(
        envelope=_envelope(
            record_id="candidate-1",
            record_type="claim_candidate",
            authority_role="candidate",
            source_records=("claim-1", "relation-1"),
        ),
        candidate_id="candidate-1",
        identity=_claim_identity(),
        value=_text("dark mode"),
        qualifiers=(_qualifier("context", "desktop"),),
        evidence_relation_ids=("relation-1",),
        proposed_action="add",
        candidate_state="submitted",
        extractor_component_id="phase2-read-adapter-challenger",
        extractor_version="1.0.0",
        model_or_rule_id="synthetic-closeout-rule",
        confidence=0.9,
        request_digest=REQUEST_DIGEST,
    )
    return ShadowAdjudicationSubmission.create(
        candidate=candidate,
        evidence_relations=(relation,),
        adjudication_envelope=_envelope(
            record_id="adjudication-1",
            record_type="adjudication_record",
            authority_role="evaluation_artifact",
            source_records=(
                "candidate-1",
                "claim-1",
                "relation-1",
            ),
        ),
        conflict_envelope=None,
        request_digest=REQUEST_DIGEST,
        expected_current_adjudication_id=None,
    )


def _episode(full_content: Mapping[str, object]) -> EpisodeRecord:
    summary = {
        "summary": "Owner selected dark mode in a synthetic closeout fixture."
    }
    return EpisodeRecord.create(
        envelope=_envelope(
            record_id="episode-1",
            record_type="episode",
            authority_role="registered_projection",
            source_records=(
                "evidence-1",
                "claim-version-1",
                "candidate-1",
            ),
            generation=1,
        ),
        episode_id="episode-1",
        episode_kind="interaction",
        episode_state="accepted",
        member_evidence_ids=("evidence-1",),
        member_claim_version_ids=("claim-version-1",),
        member_candidate_ids=("candidate-1",),
        mission_node_ids=("mission-memory",),
        participant_ids=("owner-primary",),
        valid_from=EVALUATED_AT,
        valid_to=None,
        formed_at=LATER,
        formation_component_id="m2-closeout-evaluator",
        formation_version="1.0.0",
        summary_content_digest=canonical_sha256(summary),
        full_content_digest=canonical_sha256(
            dict(full_content)
        ),
        confidence=0.9,
        supersedes_episode_id=None,
        generation=1,
    )


def _projection(
    full_content: Mapping[str, object],
) -> ProjectionVersion:
    return ProjectionVersion.create(
        envelope=_envelope(
            record_id="owner-projection-v1",
            record_type="projection_version",
            authority_role="registered_projection",
            source_records=(
                "episode-1",
                "claim-version-1",
            ),
            generation=1,
        ),
        projection_id="owner-model-1",
        version_id="owner-projection-v1",
        projection_type="owner_model",
        subject_type="owner",
        subject_id="owner-primary",
        modalities=("graph", "symbolic", "temporal", "vector"),
        generation=1,
        source_episode_ids=("episode-1",),
        source_claim_version_ids=("claim-version-1",),
        source_evidence_ids=("evidence-1",),
        valid_from=EVALUATED_AT,
        valid_to=None,
        produced_at=LATER,
        projection_state="shadow",
        responsible_component="m2-closeout-evaluator",
        model_id="synthetic-challenger",
        model_version="1.0.0",
        content_digest=canonical_sha256(
            dict(full_content)
        ),
        vector_space_id="owner-model-space",
        graph_namespace_id="owner-model-graph",
        supersedes_version_id=None,
        confidence=0.8,
    )


def _graph_edge() -> ProjectionGraphEdge:
    return ProjectionGraphEdge.create(
        edge_id="edge-1",
        graph_namespace_id="owner-model-graph",
        source_node_id="owner-primary",
        relation_type="prefers",
        target_node_id="interface-dark-mode",
        valid_from=EVALUATED_AT,
        valid_to=None,
        weight=0.9,
        source_record_ids=(
            "claim-version-1",
            "episode-1",
        ),
    )


def _evaluation_envelope(
    *,
    record_id: str,
    record_type: str,
    source_records: tuple[str, ...],
) -> MemoryUnitEnvelope:
    return _envelope(
        record_id=record_id,
        record_type=record_type,
        authority_role="evaluation_artifact",
        source_records=source_records,
        generation=1,
    )


def run_m2_closeout_evaluation(
    *,
    workspace: Path,
    report_path: Path | None = None,
) -> tuple[
    M2CloseoutReport,
    ShadowMigrationAdmissionDecision,
    dict[str, object],
]:
    """Run a deterministic synthetic M2-CLOSEOUT integration evaluation."""

    root = Path(workspace).resolve()
    private_root = root / "private-stores"
    public_root = root / "public-repository-sentinel"
    private_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)

    scope = _scope()
    namespace = _namespace()
    details: dict[str, dict[str, object]] = {}
    record_ids: list[str] = []

    foundation_envelope = _envelope(
        record_id="foundation-check-1",
        record_type="m2_foundation_check",
        authority_role="evaluation_artifact",
        source_records=("store-registration", "evidence-binding"),
    )
    foundation_envelope.validate()
    details["foundation_contracts"] = {
        "scope_digest": scope.scope_sha256(),
        "envelope_record_id": foundation_envelope.record_id,
        "status": "passed",
    }
    record_ids.append(foundation_envelope.record_id)

    claim_path = private_root / "claim-authority.sqlite3"
    with open_claim_authority_prototype(
        claim_path,
        scope=scope,
        authority_namespace_id=namespace,
        created_at=EVALUATED_AT,
        repository_root=public_root,
    ) as claim_store:
        claim_receipt = claim_store.append(
            _claim_request(),
            committed_at=LATER,
        )
        claim_current = claim_store.load_current("claim-1")
        claim_integrity = claim_store.verify_integrity()
        if not claim_integrity.valid or claim_current is None:
            raise RuntimeError(
                "Claim Authority closeout evaluation failed"
            )
        details["claim_authority"] = {
            "claim_id": claim_receipt.claim_id,
            "claim_version_id": claim_receipt.claim_version_id,
            "projection_id": claim_receipt.projection_id,
            "store_sequence": claim_receipt.store_sequence,
            "integrity_valid": claim_integrity.valid,
        }
        record_ids.extend(
            (
                claim_receipt.claim_id,
                claim_receipt.claim_version_id,
                claim_receipt.projection_id,
            )
        )

    shadow_path = private_root / "shadow-adjudication.sqlite3"
    with open_shadow_adjudication_prototype(
        shadow_path,
        scope=scope,
        authority_namespace_id=namespace,
        prototype_id="m2-closeout-shadow",
        created_at=EVALUATED_AT,
        repository_root=public_root,
    ) as shadow_store:
        shadow_receipt = shadow_store.submit_and_adjudicate(
            _shadow_submission(),
            decided_at=LATER,
        )
        shadow_integrity = shadow_store.verify_integrity()
        if (
            shadow_integrity.integrity_state != "verified"
            or shadow_receipt.canonical_claim_written
        ):
            raise RuntimeError(
                "Shadow adjudication closeout evaluation failed"
            )
        details["shadow_adjudication"] = {
            "candidate_id": shadow_receipt.candidate_id,
            "adjudication_id": shadow_receipt.adjudication_id,
            "outcome": shadow_receipt.outcome,
            "candidate_state": shadow_receipt.candidate_state,
            "canonical_claim_written": (
                shadow_receipt.canonical_claim_written
            ),
            "integrity_state": (
                shadow_integrity.integrity_state
            ),
        }
        record_ids.extend(
            (
                shadow_receipt.candidate_id,
                shadow_receipt.adjudication_id,
            )
        )

    episode_content = {
        "events": [
            {
                "id": "evidence-1",
                "text": "owner chose dark mode",
            }
        ],
        "claim_version_id": claim_receipt.claim_version_id,
        "adjudication_id": shadow_receipt.adjudication_id,
    }
    episode_summary = {
        "summary": "Owner selected dark mode in a synthetic closeout fixture."
    }
    projection_content = {
        "subject": "owner-primary",
        "preferences": {"interface_theme": "dark"},
        "lineage": {
            "claim_version_id": claim_receipt.claim_version_id,
            "adjudication_id": shadow_receipt.adjudication_id,
            "episode_id": "episode-1",
        },
    }
    projection_path = private_root / "projection.sqlite3"
    projection_profile = ProjectionPrototypeProfile.create(
        scope=scope,
        authority_namespace_id=namespace,
        store_id="m2-closeout-projection",
    )
    with open_projection_prototype(
        projection_path,
        profile=projection_profile,
        repository_root=public_root,
    ) as projection_store:
        episode_receipt = projection_store.append_episode(
            _episode(episode_content),
            full_content=episode_content,
            summary_content=episode_summary,
            idempotency_namespace="m2_closeout",
            idempotency_key="episode-1",
            request_digest=PROJECTION_REQUEST_DIGEST,
        )
        projection_receipt = projection_store.append_projection(
            _projection(projection_content),
            full_content=projection_content,
            idempotency_namespace="m2_closeout",
            idempotency_key="owner-projection-v1",
            request_digest=PROJECTION_REQUEST_DIGEST,
            expected_current_generation=None,
            vector=(1.0, 0.0, 0.5),
            graph_edges=(_graph_edge(),),
        )
        projection_integrity = projection_store.verify_integrity()
        projection_current = (
            projection_store.get_current_projection("owner-model-1")
        )
        neighbors = projection_store.neighbors(
            graph_namespace_id="owner-model-graph",
            node_id="owner-primary",
            relation_type="prefers",
            valid_at=LATER,
        )
        similarity = projection_store.similarity_search(
            (1.0, 0.0, 0.5),
            vector_space_id="owner-model-space",
            limit=5,
        )
        if (
            not projection_integrity.valid
            or projection_current is None
            or not neighbors
            or not similarity
        ):
            raise RuntimeError(
                "Projection closeout evaluation failed"
            )
        details["projection_fabric"] = {
            "episode_id": episode_receipt.record_id,
            "projection_version_id": projection_receipt.record_id,
            "graph_neighbor_count": len(neighbors),
            "vector_result_count": len(similarity),
            "integrity_valid": projection_integrity.valid,
        }
        record_ids.extend(
            (
                episode_receipt.record_id,
                projection_receipt.record_id,
            )
        )

    serving_path = private_root / "serving.sqlite3"
    serving_profile = BoundedServingProfile.create(
        scope=scope,
        authority_namespace_id=namespace,
        profile_id="m2-closeout-serving",
        item_budget=4,
        byte_budget=8192,
        expansion_depth=2,
        fusion_strategy="weighted_score",
        lexical_weight=1.0,
        vector_weight=1.0,
        graph_weight=1.0,
        mission_weight=0.5,
        freshness_weight=0.25,
        fallback_mode="bounded_scan",
        production_influence=False,
    )
    with open_bounded_serving_prototype(
        path=serving_path,
        profile=serving_profile,
        repository_root=public_root,
    ) as serving_store:
        serving_store.upsert_document(
            record_id="claim-1",
            record_version_id=claim_receipt.claim_version_id,
            source_kind="claim",
            authority_namespace_id=namespace,
            searchable_text="owner prefers dark mode interface",
            full_content={
                "claim": claim_current,
                "lineage": ["evidence-1", "adjudication-1"],
            },
            embedding=(1.0, 0.0, 0.5),
            graph_neighbors=("episode-1", "owner-model-1"),
            mission_node_ids=("mission-memory",),
            valid_from=EVALUATED_AT,
            valid_to=None,
            generation=1,
            updated_at=LATER,
        )
        serving_store.upsert_document(
            record_id="episode-1",
            record_version_id="episode-1-v1",
            source_kind="episode",
            authority_namespace_id=namespace,
            searchable_text="owner selected dark mode",
            full_content=episode_content,
            embedding=(0.9, 0.0, 0.5),
            graph_neighbors=("claim-1", "owner-model-1"),
            mission_node_ids=("mission-memory",),
            valid_from=EVALUATED_AT,
            valid_to=None,
            generation=1,
            updated_at=LATER,
        )
        serving_store.upsert_document(
            record_id="owner-model-1",
            record_version_id="owner-projection-v1",
            source_kind="projection",
            authority_namespace_id=namespace,
            searchable_text="owner model interface preference dark",
            full_content=projection_content,
            embedding=(1.0, 0.0, 0.5),
            graph_neighbors=("claim-1", "episode-1"),
            mission_node_ids=("mission-memory",),
            valid_from=EVALUATED_AT,
            valid_to=None,
            generation=1,
            updated_at=LATER,
        )
        for index_kind in (
            "lexical",
            "vector",
            "graph",
            "temporal",
        ):
            serving_store.set_index_state(
                index_kind=index_kind,
                generation=1,
                state="ready",
                updated_at=LATER,
            )
        serving_receipt = serving_store.serve(
            request_id="shadow-read-1",
            query_text="what interface does the owner prefer",
            query_embedding=(1.0, 0.0, 0.5),
            seed_record_ids=("claim-1",),
            mission_node_ids=("mission-memory",),
            idempotency_namespace="m2_closeout",
            idempotency_key="shadow-read-1",
            now=FINAL,
        )
        serving_integrity = serving_store.require_integrity()
        if (
            not serving_integrity.healthy
            or serving_receipt.packet.hydrated_item_count < 1
            or serving_receipt.packet.degraded
        ):
            raise RuntimeError(
                "Bounded serving closeout evaluation failed"
            )
        details["bounded_serving"] = {
            "packet_id": serving_receipt.packet.packet_id,
            "trace_id": serving_receipt.trace.trace_id,
            "hydrated_item_count": (
                serving_receipt.packet.hydrated_item_count
            ),
            "fallback_used": serving_receipt.fallback_used,
            "stale_index_observed": (
                serving_receipt.stale_index_observed
            ),
            "integrity_healthy": serving_integrity.healthy,
        }
        record_ids.extend(
            (
                serving_receipt.packet.packet_id,
                serving_receipt.trace.trace_id,
            )
        )

    workflow_path = private_root / "workflow.sqlite3"
    workflow_profile = DurableWorkflowProfile.create(
        scope=scope,
        authority_namespace_id=namespace,
        profile_id="m2-closeout-workflow",
        default_max_attempts=3,
        production_influence=False,
        canonical_claim_authority=False,
    )
    with open_durable_workflow_prototype(
        path=workflow_path,
        profile=workflow_profile,
        repository_root=public_root,
    ) as workflow_store:
        workflow_store.create_workflow(
            workflow_id="shadow-migration-review-1",
            workflow_kind="migration",
            task_id="adapter-evaluation-task-1",
            task_kind="migration",
            full_workflow_content={
                "purpose": "evaluate read-only Phase 2 adapter admission",
                "packet_id": serving_receipt.packet.packet_id,
            },
            full_task_content={
                "instruction": "compare synthetic Phase 2 and successor reads",
                "trace_id": serving_receipt.trace.trace_id,
            },
            target_record_ids=("claim-1", "owner-model-1"),
            source_record_ids=(
                "evidence-1",
                claim_receipt.claim_version_id,
                shadow_receipt.adjudication_id,
            ),
            priority=70,
            max_attempts=3,
            now=EVALUATED_AT,
            idempotency_namespace="m2_closeout",
            idempotency_key="workflow-create-1",
        )
        workflow_store.start_task(
            workflow_id="shadow-migration-review-1",
            task_id="adapter-evaluation-task-1",
            expected_generation=0,
            now=LATER,
            lease_expires_at=FINAL,
            idempotency_namespace="m2_closeout",
            idempotency_key="workflow-start-1",
            full_event_content={
                "worker": "m2-closeout-evaluator",
                "mode": "synthetic_read_only",
            },
        )
        workflow_result = workflow_store.complete_task(
            workflow_id="shadow-migration-review-1",
            task_id="adapter-evaluation-task-1",
            expected_generation=1,
            now=FINAL,
            idempotency_namespace="m2_closeout",
            idempotency_key="workflow-complete-1",
            full_event_content={
                "activity": "cross-prototype integration passed"
            },
            full_result_content={
                "claim_version_id": claim_receipt.claim_version_id,
                "adjudication_id": shadow_receipt.adjudication_id,
                "projection_version_id": (
                    projection_receipt.record_id
                ),
                "packet_id": serving_receipt.packet.packet_id,
                "trace_id": serving_receipt.trace.trace_id,
            },
        )
        workflow_integrity = workflow_store.require_integrity()
        if (
            not workflow_integrity.healthy
            or workflow_result.curation_receipt is None
            or workflow_result.curation_receipt.outcome != "completed"
        ):
            raise RuntimeError(
                "Durable workflow closeout evaluation failed"
            )
        curation_receipt: CurationReceipt = (
            workflow_result.curation_receipt
        )
        details["durable_workflow"] = {
            "workflow_id": workflow_result.workflow.workflow_id,
            "task_id": workflow_result.task.task_id,
            "curation_receipt_id": curation_receipt.receipt_id,
            "outcome": curation_receipt.outcome,
            "integrity_healthy": workflow_integrity.healthy,
        }
        record_ids.extend(
            (
                workflow_result.workflow.workflow_id,
                workflow_result.task.task_id,
                curation_receipt.receipt_id,
            )
        )

    deletion_path = private_root / "deletion.sqlite3"
    deletion_profile = DeletionPropagationProfile.create(
        scope=scope,
        authority_namespace_id=namespace,
        profile_id="m2-closeout-deletion",
        required_plane_kinds=(
            "claim_authority",
            "episode_projection",
            "retrieval_serving",
            "workflow_state",
        ),
        production_influence=False,
        canonical_claim_authority=False,
        destructive_live_deletion=False,
    )
    with open_deletion_propagation_prototype(
        path=deletion_path,
        profile=deletion_profile,
        repository_root=public_root,
    ) as deletion_store:
        deletion_store.begin_request(
            request_id="deletion-rehearsal-1",
            target_record_ids=("claim-1",),
            deletion_mode="mixed",
            reason_code="m2_closeout_rehearsal",
            authority_decision_id="closeout-authority-1",
            requested_by="m2-closeout-evaluator",
            requested_at=EVALUATED_AT,
            full_request_content={
                "target_lineage": list(record_ids),
                "mode": "synthetic_rehearsal",
            },
            idempotency_namespace="m2_closeout",
            idempotency_key="deletion-begin-1",
        )
        plane_specs = (
            ("claim_authority", "claim-authority-prototype"),
            ("episode_projection", "projection-prototype"),
            ("retrieval_serving", "bounded-serving-prototype"),
            ("workflow_state", "durable-workflow-prototype"),
        )
        generation = 1
        for index, (plane_kind, component_id) in enumerate(
            plane_specs,
            start=1,
        ):
            deletion_store.record_plane_result(
                request_id="deletion-rehearsal-1",
                plane_kind=plane_kind,
                component_id=component_id,
                deletion_mode="tombstone",
                state="completed",
                completed_at=FINAL,
                target_count=1,
                deleted_count=1,
                blocked_count=0,
                evidence_record_ids=(
                    f"deletion-evidence-{index}",
                ),
                error_code=None,
                full_result_content={
                    "plane_kind": plane_kind,
                    "target_record_id": "claim-1",
                    "rehearsal": True,
                },
                expected_generation=generation,
                idempotency_namespace="m2_closeout",
                idempotency_key=f"deletion-plane-{index}",
            )
            generation += 1
        restore_decision = deletion_store.evaluate_restore_filter(
            request_id="deletion-rehearsal-1",
            target_record_id="claim-1",
            source_snapshot_id="synthetic-snapshot-1",
            action="exclude",
            reason_code="active_deletion_receipt",
            evaluated_at=FINAL,
            source_content={
                "record_id": "claim-1",
                "synthetic": True,
            },
            replacement_record_id=None,
            full_decision_content={
                "restore_filter": "exclude active deletion target"
            },
            idempotency_namespace="m2_closeout",
            idempotency_key="restore-filter-1",
        )
        deletion_store.record_rehearsal(
            request_id="deletion-rehearsal-1",
            rehearsal_kind="rollback",
            outcome="rehearsed",
            evaluated_at=FINAL,
            affected_artifact_ids=tuple(record_ids),
            measurements={"recovery_seconds": 1},
            full_rehearsal_content={
                "result": "known-good synthetic state recoverable"
            },
            idempotency_namespace="m2_closeout",
            idempotency_key="rollback-rehearsal-1",
        )
        deletion_store.record_rehearsal(
            request_id="deletion-rehearsal-1",
            rehearsal_kind="model_retirement",
            outcome="not_applicable",
            evaluated_at=FINAL,
            affected_artifact_ids=("synthetic-challenger",),
            measurements={"residual_influence": 0.0},
            full_rehearsal_content={
                "reason": "no trained model artifact in synthetic fixture"
            },
            idempotency_namespace="m2_closeout",
            idempotency_key="retirement-rehearsal-1",
        )
        deletion_final = deletion_store.finalize(
            request_id="deletion-rehearsal-1",
            propagation_state="completed",
            effective_at=FINAL,
            rollback_state="rehearsed",
            retirement_state="not_applicable",
            full_receipt_content={
                "summary": "synthetic cross-plane deletion rehearsal passed",
                "restore_filter_decision_id": (
                    restore_decision.result_record_id
                ),
            },
            expected_generation=generation,
            completed_at=FINAL,
            idempotency_namespace="m2_closeout",
            idempotency_key="deletion-finalize-1",
        )
        deletion_integrity = deletion_store.verify_integrity()
        if (
            not deletion_integrity.healthy
            or deletion_final.generation != generation + 1
        ):
            raise RuntimeError(
                "Deletion propagation closeout evaluation failed"
            )
        details["deletion_propagation"] = {
            "request_id": deletion_final.request_id,
            "receipt_id": deletion_final.result_record_id,
            "required_plane_count": len(plane_specs),
            "restore_filter_decision_id": (
                restore_decision.result_record_id
            ),
            "integrity_healthy": deletion_integrity.healthy,
        }
        record_ids.extend(
            (
                deletion_final.request_id,
                deletion_final.result_record_id,
                restore_decision.result_record_id,
            )
        )

    lineage = {
        "evidence_id": "evidence-1",
        "claim_version_id": claim_receipt.claim_version_id,
        "candidate_id": shadow_receipt.candidate_id,
        "adjudication_id": shadow_receipt.adjudication_id,
        "episode_id": episode_receipt.record_id,
        "projection_version_id": projection_receipt.record_id,
        "context_packet_id": serving_receipt.packet.packet_id,
        "retrieval_trace_id": serving_receipt.trace.trace_id,
        "workflow_id": workflow_result.workflow.workflow_id,
        "curation_receipt_id": curation_receipt.receipt_id,
        "deletion_receipt_id": deletion_final.result_record_id,
    }
    if len(set(lineage.values())) != len(lineage):
        raise RuntimeError(
            "Cross-prototype lineage contains duplicate semantic IDs"
        )
    details["cross_prototype_lineage"] = {
        "lineage": lineage,
        "all_components_healthy": True,
        "canonical_authority_transfer": False,
        "production_influence": False,
        "private_payload_read": False,
        "phase2_migration_started": False,
    }
    record_ids.extend(lineage.values())

    component_results = tuple(
        M2ComponentEvaluation.create(
            component_id=component_id,
            state="passed",
            evidence_record_ids=(
                tuple(
                    value
                    for value in details[component_id].values()
                    if isinstance(value, str)
                    and value
                    and value.replace("-", "").replace("_", "").isalnum()
                )
                or (f"{component_id}-evidence",)
            ),
            evaluated_at=FINAL,
            details_content=details[component_id],
        )
        for component_id in sorted(details)
    )

    full_report_content = {
        "schema_version": "1.0.0",
        "evaluation_mode": "synthetic_cross_prototype",
        "component_details": details,
        "lineage": lineage,
        "activation_boundaries": {
            "production_influence": False,
            "canonical_authority_transfer": False,
            "private_payload_read": False,
            "phase2_migration_started": False,
            "p5_1e_unblocked": False,
        },
    }
    report = M2CloseoutReport.create(
        envelope=_evaluation_envelope(
            record_id="m2-closeout-evaluation-1",
            record_type="m2_closeout_report",
            source_records=tuple(sorted(set(record_ids))),
        ),
        evaluation_id="m2-closeout-evaluation-1",
        component_results=component_results,
        cross_prototype_record_ids=tuple(
            sorted(set(record_ids))
        ),
        evaluated_at=FINAL,
        full_report_content=full_report_content,
        production_influence=False,
        canonical_authority_transfer=False,
        private_payload_read=False,
        phase2_migration_started=False,
    )
    decision = ShadowMigrationAdmissionDecision.create(
        envelope=_evaluation_envelope(
            record_id="shadow-migration-admission-1",
            record_type="shadow_migration_admission",
            source_records=(
                report.evaluation_id,
                report.report_sha256,
            ),
        ),
        decision_id="shadow-migration-admission-1",
        evaluation_id=report.evaluation_id,
        outcome="admitted_preparatory_read_only",
        admitted_stages=ADMITTED_STAGES,
        excluded_stages=EXCLUDED_STAGES,
        reason_codes=(
            "m2_contract_sequence_complete",
            "synthetic_cross_prototype_integration_passed",
            "read_only_preparation_only",
            "production_activation_not_reviewed",
            "private_payload_migration_not_reviewed",
        ),
        decided_at=FINAL,
        phase2_migration_started=False,
        p5_1e_unblocked=False,
        production_write_mirroring=False,
        canonical_authority_transfer=False,
        private_payload_read=False,
    )
    decision.assert_supported_by(report)

    payload = {
        "schema_version": "1.0.0",
        "m2_closeout_report": report.metadata_record(),
        "shadow_migration_admission": decision.metadata_record(),
        "full_evaluation_content": full_report_content,
        "admission_scope": {
            "admitted_stages": list(decision.admitted_stages),
            "excluded_stages": list(decision.excluded_stages),
            "next_gate": (
                "implement Stage A inventory and registration plus "
                "Stage B read-only contract adapters"
            ),
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)

    if report_path is not None:
        destination = Path(report_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return report, decision, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, decision, payload = run_m2_closeout_evaluation(
        workspace=args.workspace,
        report_path=args.report,
    )
    print(f"m2_closeout_evaluation_id={report.evaluation_id}")
    print(f"m2_closeout_report_sha256={report.report_sha256}")
    print(f"shadow_migration_admission={decision.outcome}")
    print(f"phase2_migration_started={decision.phase2_migration_started}")
    print(f"p5_1e_unblocked={decision.p5_1e_unblocked}")
    print(f"artifact_sha256={payload['artifact_sha256']}")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
