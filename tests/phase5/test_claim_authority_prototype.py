"""Memory M2.1 reversible Claim Authority prototype tests."""

from __future__ import annotations

from dataclasses import replace
import sqlite3
from pathlib import Path

import pytest

from cognitive_kernel.claim_authority_prototype import (
    CLAIM_AUTHORITY_PROTOTYPE_STATE,
    ClaimAuthorityAppendRequest,
    ClaimAuthorityPrototypeConflictError,
    ClaimAuthorityPrototypeIntegrityError,
    ClaimAuthorityPrototypeIsolationError,
    UnsafeClaimAuthorityPrototypePathError,
    open_claim_authority_prototype,
)
from cognitive_kernel.claim_contracts import (
    CanonicalTaggedValue,
    ClaimIdentity,
    ClaimQualifier,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope

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
    supersedes: tuple[str, ...] = (),
    product_scope: ProductHostScope | None = None,
    workflow_id: str = "request-1",
) -> MemoryUnitEnvelope:
    selected_scope = product_scope or scope()
    return MemoryUnitEnvelope.create(
        scope=selected_scope,
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id=(
            "owner-primary"
            if selected_scope.product_id == "alice"
            else "synthetic-host-a"
        ),
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
        writer="claim_authority",
        workflow_or_request_id=workflow_id,
        idempotency_namespace="claim_authority",
        idempotency_key=record_id,
        supersedes=supersedes,
    )


def text(value: str) -> CanonicalTaggedValue:
    return CanonicalTaggedValue.create(type_tag="text", value=value)


def qualifier(key: str, value: str) -> ClaimQualifier:
    return ClaimQualifier.create(key=key, value=text(value))


def identity(
    *,
    product_scope: ProductHostScope | None = None,
) -> ClaimIdentity:
    return ClaimIdentity.create(
        envelope=envelope(
            record_id="claim-1",
            record_type="claim_identity",
            authority_role="claim_authority",
            product_scope=product_scope,
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


def request(
    *,
    version_id: str = "claim-version-1",
    projection_id: str = "projection-1",
    value: str = "dark mode",
    request_digest: str = REQUEST_DIGEST,
    expected_current: str | None = None,
    supersedes: tuple[str, ...] = (),
    correction_of: tuple[str, ...] = (),
    product_scope: ProductHostScope | None = None,
) -> ClaimAuthorityAppendRequest:
    selected_scope = product_scope or scope()
    claim = identity(product_scope=selected_scope)
    return ClaimAuthorityAppendRequest.create(
        identity=claim,
        version_envelope=envelope(
            record_id=version_id,
            record_type="claim_version",
            authority_role="claim_authority",
            source_records=(claim.claim_id, "evidence-1"),
            supersedes=supersedes,
            product_scope=selected_scope,
            workflow_id=version_id,
        ),
        projection_envelope=envelope(
            record_id=projection_id,
            record_type="current_claim_projection",
            authority_role="registered_projection",
            source_records=(claim.claim_id, version_id),
            product_scope=selected_scope,
            workflow_id=projection_id,
        ),
        value=text(value),
        qualifiers=(qualifier("context", "desktop"),),
        authority_class="owner_attested",
        confidence=0.99,
        adjudication_state="accepted",
        evidence_relation_ids=("binding-1",),
        correction_of=correction_of,
        request_digest=request_digest,
        expected_current_claim_version_id=expected_current,
        validity_state="current",
        conflict_state="none",
        deletion_state="active",
    )


def open_store(path: Path, *, selected_scope: ProductHostScope | None = None):
    repository_root = path.parents[1] / "public-repo"
    repository_root.mkdir(parents=True, exist_ok=True)
    chosen_scope = selected_scope or scope()
    return open_claim_authority_prototype(
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
    with pytest.raises(UnsafeClaimAuthorityPrototypePathError):
        open_claim_authority_prototype(
            repository / "claim.db",
            scope=scope(),
            authority_namespace_id="owner-primary",
            created_at=REFERENCE_TIME,
            repository_root=repository,
        )


def test_append_persists_full_canonical_claim_content(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        receipt = store.append(request(), committed_at=REFERENCE_TIME)
        history = store.history("claim-1")
        current = store.load_current("claim-1")
        report = store.verify_integrity()

    assert receipt.store_sequence == 1
    assert receipt.version_sequence == 1
    assert receipt.idempotent_replay is False
    assert history[0]["value"]["value"] == "dark mode"
    assert current["current_claim_version_id"] == "claim-version-1"
    assert report.identity_count == 1
    assert report.version_count == 1
    assert report.current_count == 1
    assert report.valid is True


def test_reopen_preserves_current_state_without_history_replay(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        store.append(request(), committed_at=REFERENCE_TIME)
    with open_store(database) as reopened:
        current = reopened.load_current("claim-1")
        report = reopened.verify_integrity()

    assert current["source_position"] == 1
    assert report.last_store_sequence == 1
    assert reopened.prototype_state == CLAIM_AUTHORITY_PROTOTYPE_STATE


def test_idempotent_retry_returns_prior_receipt(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    value = request()
    with open_store(database) as store:
        first = store.append(value, committed_at=REFERENCE_TIME)
        replay = store.append(value, committed_at=LATER_TIME)
        report = store.verify_integrity()

    assert replay.idempotent_replay is True
    assert replay.store_sequence == first.store_sequence
    assert replay.claim_version_id == first.claim_version_id
    assert report.version_count == 1


def test_idempotency_key_reuse_with_new_digest_is_conflict(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        store.append(request(), committed_at=REFERENCE_TIME)
        changed = request(request_digest=SECOND_REQUEST_DIGEST)
        with pytest.raises(ClaimAuthorityPrototypeConflictError):
            store.append(changed, committed_at=LATER_TIME)


def test_store_assigns_global_and_per_claim_sequences(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        first = store.append(request(), committed_at=REFERENCE_TIME)
        second = store.append(
            request(
                version_id="claim-version-2",
                projection_id="projection-2",
                value="system mode",
                request_digest=SECOND_REQUEST_DIGEST,
                expected_current="claim-version-1",
                supersedes=("claim-version-1",),
                correction_of=("claim-version-1",),
            ),
            committed_at=LATER_TIME,
        )
        history = store.history("claim-1")
        current = store.load_current("claim-1")

    assert (first.store_sequence, second.store_sequence) == (1, 2)
    assert (first.version_sequence, second.version_sequence) == (1, 2)
    assert len(history) == 2
    assert current["current_claim_version_id"] == "claim-version-2"
    assert current["projection_generation"] == 2


def test_expected_current_version_prevents_lost_update(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        store.append(request(), committed_at=REFERENCE_TIME)
        stale = request(
            version_id="claim-version-2",
            projection_id="projection-2",
            request_digest=SECOND_REQUEST_DIGEST,
            expected_current="claim-version-missing",
        )
        with pytest.raises(ClaimAuthorityPrototypeConflictError):
            store.append(stale, committed_at=LATER_TIME)
        assert store.verify_integrity().version_count == 1


def test_store_rejects_cross_product_request(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="synthetic-private",
    )
    with open_store(database) as store:
        with pytest.raises(ClaimAuthorityPrototypeIsolationError):
            store.append(
                request(product_scope=friday_scope),
                committed_at=REFERENCE_TIME,
            )


def test_reopen_rejects_scope_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database):
        pass
    friday_scope = scope(
        product_id="friday",
        host_instance_id="synthetic-host-a",
        encryption_domain="synthetic-private",
    )
    with pytest.raises(ClaimAuthorityPrototypeIsolationError):
        open_store(database, selected_scope=friday_scope)


def test_claim_versions_are_sqlite_append_only(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        store.append(request(), committed_at=REFERENCE_TIME)
        with pytest.raises(sqlite3.DatabaseError):
            store._connection.execute(
                "UPDATE claim_versions SET request_digest = ? "
                "WHERE store_sequence = 1",
                (SECOND_REQUEST_DIGEST,),
            )
        store._connection.rollback()
        with pytest.raises(sqlite3.DatabaseError):
            store._connection.execute(
                "DELETE FROM claim_versions WHERE store_sequence = 1"
            )
        store._connection.rollback()


def test_integrity_detects_current_projection_tampering(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        store.append(request(), committed_at=REFERENCE_TIME)
        store._connection.execute(
            "UPDATE current_claims SET projection_json = ? WHERE claim_id = ?",
            ('{"broken":true}', "claim-1"),
        )
        store._connection.commit()
        with pytest.raises(ClaimAuthorityPrototypeIntegrityError):
            store.verify_integrity()


def test_prototype_state_never_claims_production_authority(tmp_path: Path) -> None:
    database = tmp_path / "vault" / "claim.db"
    with open_store(database) as store:
        row = store._connection.execute(
            "SELECT prototype_state FROM claim_authority_metadata "
            "WHERE singleton = 1"
        ).fetchone()
    assert row["prototype_state"] == "reversible_nonproduction"
    assert "production" not in {"enabled", row["prototype_state"]}
