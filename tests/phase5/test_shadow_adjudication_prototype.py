"""Memory M2.2 reversible shadow-adjudication prototype tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from cognitive_kernel.adjudication_contracts import (
    ClaimCandidate,
    ClaimEvidenceRelation,
)
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.claim_contracts import (
    CanonicalTaggedValue,
    ClaimIdentity,
    ClaimQualifier,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope
from cognitive_kernel.shadow_adjudication_prototype import (
    SHADOW_ADJUDICATION_PROTOTYPE_STATE,
    ShadowAdjudicationConflictError,
    ShadowAdjudicationIntegrityError,
    ShadowAdjudicationIsolationError,
    ShadowAdjudicationProfile,
    ShadowAdjudicationSubmission,
    UnsafeShadowAdjudicationPathError,
    open_shadow_adjudication_prototype,
)

REFERENCE_TIME = "2026-08-06T04:00:00Z"
LATER_TIME = "2026-08-06T05:00:00Z"
PROVENANCE_DIGEST = "a" * 64
CONTENT_DIGEST = "b" * 64
REQUEST_DIGEST = "c" * 64
SECOND_REQUEST_DIGEST = "d" * 64


def scope(
    *,
    product_id: str = "alice",
    host_instance_id: str = "owner-primary",
    encryption_domain: str = "owner-private",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host_instance_id,
        schema_version="1.0.0",
        encryption_domain=encryption_domain,
    )


def envelope(
    *,
    record_id: str,
    record_type: str,
    authority_role: str,
    source_records: tuple[str, ...] = (),
    product_scope: ProductHostScope | None = None,
    idempotency_key: str | None = None,
) -> MemoryUnitEnvelope:
    selected_scope = product_scope or scope()
    namespace = (
        "owner-primary"
        if selected_scope.product_id == "alice"
        else "synthetic-host-a"
    )
    return MemoryUnitEnvelope.create(
        scope=selected_scope,
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id=namespace,
        host_or_cluster_id=selected_scope.host_instance_id,
        authority_role=authority_role,
        deployment_profile="single_workstation",
        created_at=REFERENCE_TIME,
        valid_from=REFERENCE_TIME,
        valid_to=None,
        transaction_time=REFERENCE_TIME,
        logical_clock=1,
        causal_parents=(),
        source_records=source_records,
        generation=0,
        state="committed",
        data_classification="highly_sensitive",
        retention_class="authoritative_source",
        deletion_state="active",
        provenance_digest=PROVENANCE_DIGEST,
        content_digest=CONTENT_DIGEST,
        writer="shadow_adjudicator",
        workflow_or_request_id="request-1",
        idempotency_namespace="shadow_adjudication",
        idempotency_key=idempotency_key or record_id,
    )


def text(value: str) -> CanonicalTaggedValue:
    return CanonicalTaggedValue.create(type_tag="text", value=value)


def qualifier(key: str, value: str) -> ClaimQualifier:
    return ClaimQualifier.create(key=key, value=text(value))


def identity(
    *, product_scope: ProductHostScope | None = None
) -> ClaimIdentity:
    selected_scope = product_scope or scope()
    return ClaimIdentity.create(
        envelope=envelope(
            record_id="claim-1",
            record_type="claim_identity",
            authority_role="claim_authority",
            product_scope=selected_scope,
        ),
        claim_id="claim-1",
        canonical_subject=CanonicalTaggedValue.create(
            type_tag="identifier",
            value="owner-primary",
        ),
        canonical_predicate="prefers_interface",
        canonical_value=text("dark mode"),
        qualifiers=(),
        semantic_scope=("user_interface", "owner_preference"),
        canonicalization_version="1.0.0",
    )


def relation(
    *,
    relation_id: str,
    evidence_id: str,
    relation_type: str,
    product_scope: ProductHostScope | None = None,
) -> ClaimEvidenceRelation:
    selected_scope = product_scope or scope()
    return ClaimEvidenceRelation.create(
        envelope=envelope(
            record_id=relation_id,
            record_type="claim_evidence_relation",
            authority_role="candidate",
            source_records=(evidence_id, "candidate-1"),
            product_scope=selected_scope,
        ),
        relation_id=relation_id,
        evidence_record_id=evidence_id,
        target_record_id="candidate-1",
        target_record_type="claim_candidate",
        relation_type=relation_type,
        source_class="experience_ledger_event",
        source_authority_class="owner_attested",
        extractor_component_id="candidate-extractor",
        extractor_version="1.0.0",
        confidence=0.9,
    )


def candidate(
    *,
    relations: tuple[ClaimEvidenceRelation, ...],
    confidence: float | None = 0.9,
    proposed_action: str = "add",
    request_digest: str = REQUEST_DIGEST,
    product_scope: ProductHostScope | None = None,
) -> ClaimCandidate:
    selected_scope = product_scope or scope()
    relation_ids = tuple(item.relation_id for item in relations)
    return ClaimCandidate.create(
        envelope=envelope(
            record_id="candidate-1",
            record_type="claim_candidate",
            authority_role="candidate",
            source_records=("claim-1", *relation_ids),
            product_scope=selected_scope,
            idempotency_key="candidate-1",
        ),
        candidate_id="candidate-1",
        identity=identity(product_scope=selected_scope),
        value=text("dark mode"),
        qualifiers=(qualifier("context", "desktop"),),
        evidence_relation_ids=relation_ids,
        proposed_action=proposed_action,
        candidate_state="submitted",
        extractor_component_id="candidate-extractor",
        extractor_version="1.0.0",
        model_or_rule_id="rule-owner-preference-v1",
        confidence=confidence,
        request_digest=request_digest,
    )


def submission(
    *,
    relation_types: tuple[str, ...] = ("support",),
    confidence: float | None = 0.9,
    proposed_action: str = "add",
    request_digest: str = REQUEST_DIGEST,
    expected_current: str | None = None,
    product_scope: ProductHostScope | None = None,
) -> ShadowAdjudicationSubmission:
    selected_scope = product_scope or scope()
    relations = tuple(
        relation(
            relation_id=f"relation-{index}",
            evidence_id=f"evidence-{index}",
            relation_type=relation_type,
            product_scope=selected_scope,
        )
        for index, relation_type in enumerate(relation_types, start=1)
    )
    value = candidate(
        relations=relations,
        confidence=confidence,
        proposed_action=proposed_action,
        request_digest=request_digest,
        product_scope=selected_scope,
    )
    relation_ids = tuple(item.relation_id for item in relations)
    evidence_ids = tuple(item.evidence_record_id for item in relations)
    has_conflict = "contradiction" in relation_types and any(
        item in relation_types for item in ("support", "derivation")
    )
    conflict_id = "conflict-1" if has_conflict else None
    adjudication_sources = [
        value.candidate_id,
        value.identity.claim_id,
        *relation_ids,
    ]
    if conflict_id is not None:
        adjudication_sources.append(conflict_id)
    conflict_envelope = None
    if conflict_id is not None:
        conflict_envelope = envelope(
            record_id=conflict_id,
            record_type="claim_conflict_record",
            authority_role="evaluation_artifact",
            source_records=(*evidence_ids, *relation_ids),
            product_scope=selected_scope,
        )
    return ShadowAdjudicationSubmission.create(
        candidate=value,
        evidence_relations=relations,
        adjudication_envelope=envelope(
            record_id="adjudication-1",
            record_type="adjudication_record",
            authority_role="evaluation_artifact",
            source_records=tuple(adjudication_sources),
            product_scope=selected_scope,
        ),
        conflict_envelope=conflict_envelope,
        request_digest=request_digest,
        expected_current_adjudication_id=expected_current,
    )


def open_store(path: Path, *, selected_scope: ProductHostScope | None = None):
    repository_root = path.parents[1] / "public-repo"
    repository_root.mkdir(parents=True, exist_ok=True)
    chosen_scope = selected_scope or scope()
    return open_shadow_adjudication_prototype(
        path,
        scope=chosen_scope,
        authority_namespace_id=(
            "owner-primary"
            if chosen_scope.product_id == "alice"
            else "synthetic-host-a"
        ),
        created_at=REFERENCE_TIME,
        repository_root=repository_root,
    )


def test_prototype_refuses_public_repository_path(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(UnsafeShadowAdjudicationPathError):
        open_shadow_adjudication_prototype(
            repository / "shadow.db",
            scope=scope(),
            authority_namespace_id="owner-primary",
            created_at=REFERENCE_TIME,
            repository_root=repository,
        )


def test_supported_candidate_is_shadow_eligible_without_canonical_write(
    tmp_path: Path,
) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        receipt = store.submit_and_adjudicate(
            submission(), decided_at=REFERENCE_TIME
        )
        current = store.load_current_state("candidate-1")
        candidate_record = store.load_candidate("candidate-1")
        report = store.verify_integrity()

    assert receipt.outcome == "add"
    assert receipt.candidate_state == "eligible"
    assert current["canonical_claim_written"] is False
    assert candidate_record["value"]["value"] == "dark mode"
    assert report.integrity_state == "verified"
    assert report.candidate_count == 1
    assert report.adjudication_count == 1


def test_insufficient_support_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        receipt = store.submit_and_adjudicate(
            submission(relation_types=()), decided_at=REFERENCE_TIME
        )
        current = store.load_current_state("candidate-1")
    assert receipt.outcome == "reject"
    assert current["candidate_state"] == "rejected"


def test_deletion_cause_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        receipt = store.submit_and_adjudicate(
            submission(relation_types=("deletion_cause",)),
            decided_at=REFERENCE_TIME,
        )
    assert receipt.outcome == "reject"


def test_support_contradiction_creates_quarantined_conflict(
    tmp_path: Path,
) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        receipt = store.submit_and_adjudicate(
            submission(relation_types=("support", "contradiction")),
            decided_at=REFERENCE_TIME,
        )
        current = store.load_current_state("candidate-1")
        report = store.verify_integrity()
    assert receipt.outcome == "quarantine"
    assert receipt.conflict_id == "conflict-1"
    assert current["candidate_state"] == "quarantined"
    assert report.conflict_count == 1


def test_low_confidence_is_quarantined(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        receipt = store.submit_and_adjudicate(
            submission(confidence=0.5), decided_at=REFERENCE_TIME
        )
    assert receipt.outcome == "quarantine"


def test_nonautomatic_action_is_quarantined(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        receipt = store.submit_and_adjudicate(
            submission(proposed_action="request_owner_input"),
            decided_at=REFERENCE_TIME,
        )
    assert receipt.outcome == "quarantine"


def test_idempotent_retry_returns_prior_receipt(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    value = submission()
    with open_store(database) as store:
        first = store.submit_and_adjudicate(
            value, decided_at=REFERENCE_TIME
        )
        replay = store.submit_and_adjudicate(
            value, decided_at=LATER_TIME
        )
        report = store.verify_integrity()
    assert replay.idempotent_replay is True
    assert replay.decision_sequence == first.decision_sequence
    assert report.adjudication_count == 1


def test_idempotency_key_reuse_with_changed_digest_is_conflict(
    tmp_path: Path,
) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        store.submit_and_adjudicate(
            submission(), decided_at=REFERENCE_TIME
        )
        with pytest.raises(ShadowAdjudicationConflictError):
            store.submit_and_adjudicate(
                submission(request_digest=SECOND_REQUEST_DIGEST),
                decided_at=LATER_TIME,
            )


def test_reopen_preserves_materialized_candidate_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        store.submit_and_adjudicate(
            submission(), decided_at=REFERENCE_TIME
        )
    with open_store(database) as reopened:
        current = reopened.load_current_state("candidate-1")
        history = reopened.decision_history("candidate-1")
        report = reopened.verify_integrity()
    assert current["decision_generation"] == 1
    assert len(history) == 1
    assert report.adjudication_count == 1
    assert reopened.prototype_state == SHADOW_ADJUDICATION_PROTOTYPE_STATE


def test_store_rejects_cross_product_submission(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="synthetic-private",
    )
    with open_store(database) as store:
        with pytest.raises(ShadowAdjudicationIsolationError):
            store.submit_and_adjudicate(
                submission(product_scope=friday_scope),
                decided_at=REFERENCE_TIME,
            )


def test_reopen_rejects_scope_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database):
        pass
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="synthetic-private",
    )
    with pytest.raises(ShadowAdjudicationIsolationError):
        open_store(database, selected_scope=friday_scope)


def test_expected_current_prevents_stale_shadow_decision(
    tmp_path: Path,
) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        with pytest.raises(ShadowAdjudicationConflictError):
            store.submit_and_adjudicate(
                submission(expected_current="missing-adjudication"),
                decided_at=REFERENCE_TIME,
            )


def test_integrity_detects_adjudication_tampering(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "shadow.db"
    with open_store(database) as store:
        store.submit_and_adjudicate(
            submission(), decided_at=REFERENCE_TIME
        )
    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER adjudication_records_no_update")
    connection.execute(
        "UPDATE adjudication_records SET adjudication_sha256 = ?",
        ("0" * 64,),
    )
    connection.commit()
    connection.close()
    with open_store(database) as store:
        with pytest.raises(ShadowAdjudicationIntegrityError):
            store.verify_integrity()


def test_submission_requires_exact_relation_set() -> None:
    value = submission()
    extra = relation(
        relation_id="relation-2",
        evidence_id="evidence-2",
        relation_type="support",
    )
    with pytest.raises(CognitiveKernelContractError):
        ShadowAdjudicationSubmission.create(
            candidate=value.candidate,
            evidence_relations=(*value.evidence_relations, extra),
            adjudication_envelope=value.adjudication_envelope,
            conflict_envelope=None,
            request_digest=REQUEST_DIGEST,
        )
