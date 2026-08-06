"""Memory M2.2 evidence, candidate, adjudication, and conflict contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    require_confidence,
    require_identifier,
    require_sha256,
)
from .claim_contracts import (
    CanonicalTaggedValue,
    ClaimIdentity,
    ClaimQualifier,
    normalize_claim_qualifiers,
)
from .contracts import ProductHostScope
from .memory_contracts import EVIDENCE_RELATION_TYPES, MemoryUnitEnvelope

ADJUDICATION_CONTRACT_SCHEMA_VERSION = "1.0.0"

ADJUDICATION_OUTCOMES = frozenset(
    {
        "add",
        "revise",
        "supersede",
        "dispute",
        "quarantine",
        "merge",
        "split",
        "reject",
    }
)

ADJUDICATION_EXECUTION_MODES = frozenset(
    {
        "shadow",
        "canary",
        "production",
    }
)

CLAIM_CANDIDATE_STATES = frozenset(
    {
        "submitted",
        "eligible",
        "rejected",
        "quarantined",
        "adjudicated",
        "withdrawn",
    }
)

CLAIM_CANDIDATE_ACTIONS = frozenset(
    {
        "add",
        "revise",
        "supersede",
        "dispute",
        "merge",
        "split",
        "retain_as_evidence",
        "request_owner_input",
        "schedule_experiment",
        "create_training_candidate",
    }
)

CLAIM_CONFLICT_TYPES = frozenset(
    {
        "support_contradiction",
        "correction",
        "temporal_overlap",
        "authority_disagreement",
        "semantic_collision",
        "replication",
        "duplicate",
        "other",
    }
)

CLAIM_CONFLICT_RESOLUTION_STATES = frozenset(
    {
        "open",
        "quarantined",
        "resolved",
        "dismissed",
        "superseded",
    }
)

ADJUDICATION_AUTHORITY_CLASSES = frozenset(
    {
        "owner",
        "human_reviewer",
        "model",
        "ensemble",
        "graph",
        "algorithmic",
        "experimental",
        "mission",
        "constitutional",
    }
)


def _sorted_identifiers(
    values: Iterable[object],
    field: str,
    *,
    minimum: int = 0,
) -> tuple[str, ...]:
    normalized = tuple(
        sorted(normalize_identifier_sequence(values, field))
    )
    if len(normalized) < minimum:
        raise CognitiveKernelContractError(
            f"{field} must contain at least {minimum} identifiers"
        )
    return normalized


def _optional_identifier(value: object | None, field: str) -> str | None:
    return None if value is None else require_identifier(value, field)


def _same_scope(first: ProductHostScope, second: ProductHostScope) -> bool:
    first.validate()
    second.validate()
    return first.metadata_record() == second.metadata_record()


def _require_envelope(
    envelope: MemoryUnitEnvelope,
    *,
    record_id: str,
    record_type: str,
    authority_roles: frozenset[str],
) -> None:
    envelope.validate()
    if envelope.record_id != record_id:
        raise CognitiveKernelContractError(
            f"{record_type} envelope record_id differs from contract id"
        )
    if envelope.record_type != record_type:
        raise CognitiveKernelContractError(
            f"envelope must use record_type {record_type}"
        )
    if envelope.authority_role not in authority_roles:
        raise CognitiveKernelContractError(
            f"{record_type} envelope authority role is not permitted"
        )


@dataclass(frozen=True)
class ClaimEvidenceRelation:
    """Append-only relation from evidence to a candidate, claim, or model."""

    envelope: MemoryUnitEnvelope
    relation_id: str
    evidence_record_id: str
    target_record_id: str
    target_record_type: str
    relation_type: str
    source_class: str
    source_authority_class: str
    extractor_component_id: str
    extractor_version: str
    confidence: float | None
    relation_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        relation_id: object,
        evidence_record_id: object,
        target_record_id: object,
        target_record_type: object,
        relation_type: object,
        source_class: object,
        source_authority_class: object,
        extractor_component_id: object,
        extractor_version: object,
        confidence: float | None,
    ) -> "ClaimEvidenceRelation":
        draft = cls(
            envelope=envelope,
            relation_id=require_identifier(relation_id, "relation_id"),
            evidence_record_id=require_identifier(
                evidence_record_id, "evidence_record_id"
            ),
            target_record_id=require_identifier(
                target_record_id, "target_record_id"
            ),
            target_record_type=require_identifier(
                target_record_type, "target_record_type"
            ),
            relation_type=require_identifier(
                relation_type, "relation_type"
            ),
            source_class=require_identifier(source_class, "source_class"),
            source_authority_class=require_identifier(
                source_authority_class, "source_authority_class"
            ),
            extractor_component_id=require_identifier(
                extractor_component_id, "extractor_component_id"
            ),
            extractor_version=require_identifier(
                extractor_version, "extractor_version"
            ),
            confidence=(
                require_confidence(confidence, "confidence")
                if confidence is not None
                else None
            ),
            relation_sha256="0" * 64,
        )
        result = cls(
            **{
                **draft.__dict__,
                "relation_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": ADJUDICATION_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "relation_id": self.relation_id,
            "evidence_record_id": self.evidence_record_id,
            "target_record_id": self.target_record_id,
            "target_record_type": self.target_record_type,
            "relation_type": self.relation_type,
            "source_class": self.source_class,
            "source_authority_class": self.source_authority_class,
            "extractor_component_id": self.extractor_component_id,
            "extractor_version": self.extractor_version,
            "confidence": self.confidence,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["relation_sha256"] = self.relation_sha256
        return record

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_id=self.relation_id,
            record_type="claim_evidence_relation",
            authority_roles=frozenset(
                {"candidate", "claim_authority", "evaluation_artifact"}
            ),
        )
        if self.relation_type not in EVIDENCE_RELATION_TYPES:
            raise CognitiveKernelContractError(
                "relation_type is not ratified"
            )
        allowed_targets = {
            "claim_candidate",
            "claim_identity",
            "claim_version",
            "model_version",
            "adjudication_record",
        }
        if self.target_record_type not in allowed_targets:
            raise CognitiveKernelContractError(
                "target_record_type is not supported"
            )
        if self.evidence_record_id not in self.envelope.source_records:
            raise CognitiveKernelContractError(
                "relation envelope must bind evidence_record_id"
            )
        if self.target_record_id not in self.envelope.source_records:
            raise CognitiveKernelContractError(
                "relation envelope must bind target_record_id"
            )
        require_sha256(self.relation_sha256, "relation_sha256")
        if canonical_sha256(self.material_record()) != self.relation_sha256:
            raise CognitiveKernelContractError(
                "evidence relation digest mismatch"
            )


@dataclass(frozen=True)
class ClaimCandidate:
    """Full-content candidate that remains distinct from adjudicated truth."""

    envelope: MemoryUnitEnvelope
    candidate_id: str
    identity: ClaimIdentity
    value: CanonicalTaggedValue
    qualifiers: tuple[ClaimQualifier, ...]
    evidence_relation_ids: tuple[str, ...]
    proposed_action: str
    candidate_state: str
    extractor_component_id: str
    extractor_version: str
    model_or_rule_id: str
    confidence: float | None
    request_digest: str
    candidate_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        candidate_id: object,
        identity: ClaimIdentity,
        value: CanonicalTaggedValue,
        qualifiers: Iterable[ClaimQualifier] = (),
        evidence_relation_ids: Iterable[object] = (),
        proposed_action: object,
        candidate_state: object = "submitted",
        extractor_component_id: object,
        extractor_version: object,
        model_or_rule_id: object,
        confidence: float | None,
        request_digest: object,
    ) -> "ClaimCandidate":
        normalized_id = require_identifier(candidate_id, "candidate_id")
        draft = cls(
            envelope=envelope,
            candidate_id=normalized_id,
            identity=identity,
            value=value,
            qualifiers=normalize_claim_qualifiers(qualifiers),
            evidence_relation_ids=_sorted_identifiers(
                evidence_relation_ids, "evidence_relation_ids"
            ),
            proposed_action=require_identifier(
                proposed_action, "proposed_action"
            ),
            candidate_state=require_identifier(
                candidate_state, "candidate_state"
            ),
            extractor_component_id=require_identifier(
                extractor_component_id, "extractor_component_id"
            ),
            extractor_version=require_identifier(
                extractor_version, "extractor_version"
            ),
            model_or_rule_id=require_identifier(
                model_or_rule_id, "model_or_rule_id"
            ),
            confidence=(
                require_confidence(confidence, "confidence")
                if confidence is not None
                else None
            ),
            request_digest=require_sha256(
                request_digest, "request_digest"
            ),
            candidate_sha256="0" * 64,
        )
        result = cls(
            **{
                **draft.__dict__,
                "candidate_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": ADJUDICATION_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "candidate_id": self.candidate_id,
            "identity": self.identity.metadata_record(),
            "value": self.value.metadata_record(),
            "qualifiers": [item.metadata_record() for item in self.qualifiers],
            "evidence_relation_ids": list(self.evidence_relation_ids),
            "proposed_action": self.proposed_action,
            "candidate_state": self.candidate_state,
            "extractor_component_id": self.extractor_component_id,
            "extractor_version": self.extractor_version,
            "model_or_rule_id": self.model_or_rule_id,
            "confidence": self.confidence,
            "request_digest": self.request_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["candidate_sha256"] = self.candidate_sha256
        return record

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_id=self.candidate_id,
            record_type="claim_candidate",
            authority_roles=frozenset({"candidate"}),
        )
        self.identity.validate()
        self.value.validate()
        normalize_claim_qualifiers(self.qualifiers)
        if not _same_scope(self.envelope.scope, self.identity.envelope.scope):
            raise CognitiveKernelContractError(
                "candidate crosses product-host-encryption scope"
            )
        if (
            self.envelope.authority_namespace_id
            != self.identity.envelope.authority_namespace_id
        ):
            raise CognitiveKernelContractError(
                "candidate crosses authority namespace"
            )
        if self.identity.claim_id not in self.envelope.source_records:
            raise CognitiveKernelContractError(
                "candidate envelope must bind claim identity"
            )
        if set(self.evidence_relation_ids) - set(
            self.envelope.source_records
        ):
            raise CognitiveKernelContractError(
                "candidate envelope must bind every evidence relation"
            )
        if self.proposed_action not in CLAIM_CANDIDATE_ACTIONS:
            raise CognitiveKernelContractError(
                "proposed_action is not ratified"
            )
        if self.candidate_state not in CLAIM_CANDIDATE_STATES:
            raise CognitiveKernelContractError(
                "candidate_state is not ratified"
            )
        require_sha256(self.request_digest, "request_digest")
        require_sha256(self.candidate_sha256, "candidate_sha256")
        if canonical_sha256(self.material_record()) != self.candidate_sha256:
            raise CognitiveKernelContractError("candidate digest mismatch")


@dataclass(frozen=True)
class ClaimConflictRecord:
    """Append-only record of a detected or resolved claim conflict."""

    envelope: MemoryUnitEnvelope
    conflict_id: str
    claim_id: str | None
    member_record_ids: tuple[str, ...]
    evidence_relation_ids: tuple[str, ...]
    conflict_type: str
    resolution_state: str
    detected_by: str
    detection_rule_id: str
    resolution_adjudication_id: str | None
    rollback_reference: str
    conflict_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        conflict_id: object,
        claim_id: object | None,
        member_record_ids: Iterable[object],
        evidence_relation_ids: Iterable[object] = (),
        conflict_type: object,
        resolution_state: object,
        detected_by: object,
        detection_rule_id: object,
        resolution_adjudication_id: object | None,
        rollback_reference: object,
    ) -> "ClaimConflictRecord":
        normalized_id = require_identifier(conflict_id, "conflict_id")
        draft = cls(
            envelope=envelope,
            conflict_id=normalized_id,
            claim_id=_optional_identifier(claim_id, "claim_id"),
            member_record_ids=_sorted_identifiers(
                member_record_ids, "member_record_ids", minimum=2
            ),
            evidence_relation_ids=_sorted_identifiers(
                evidence_relation_ids, "evidence_relation_ids"
            ),
            conflict_type=require_identifier(
                conflict_type, "conflict_type"
            ),
            resolution_state=require_identifier(
                resolution_state, "resolution_state"
            ),
            detected_by=require_identifier(detected_by, "detected_by"),
            detection_rule_id=require_identifier(
                detection_rule_id, "detection_rule_id"
            ),
            resolution_adjudication_id=_optional_identifier(
                resolution_adjudication_id,
                "resolution_adjudication_id",
            ),
            rollback_reference=require_identifier(
                rollback_reference, "rollback_reference"
            ),
            conflict_sha256="0" * 64,
        )
        result = cls(
            **{
                **draft.__dict__,
                "conflict_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": ADJUDICATION_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "conflict_id": self.conflict_id,
            "claim_id": self.claim_id,
            "member_record_ids": list(self.member_record_ids),
            "evidence_relation_ids": list(self.evidence_relation_ids),
            "conflict_type": self.conflict_type,
            "resolution_state": self.resolution_state,
            "detected_by": self.detected_by,
            "detection_rule_id": self.detection_rule_id,
            "resolution_adjudication_id": (
                self.resolution_adjudication_id
            ),
            "rollback_reference": self.rollback_reference,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["conflict_sha256"] = self.conflict_sha256
        return record

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_id=self.conflict_id,
            record_type="claim_conflict_record",
            authority_roles=frozenset(
                {"candidate", "claim_authority", "evaluation_artifact"}
            ),
        )
        if self.conflict_type not in CLAIM_CONFLICT_TYPES:
            raise CognitiveKernelContractError(
                "conflict_type is not ratified"
            )
        if self.resolution_state not in CLAIM_CONFLICT_RESOLUTION_STATES:
            raise CognitiveKernelContractError(
                "resolution_state is not ratified"
            )
        if self.resolution_state in {
            "resolved",
            "dismissed",
            "superseded",
        } and self.resolution_adjudication_id is None:
            raise CognitiveKernelContractError(
                "resolved conflict requires resolution adjudication"
            )
        if set(self.member_record_ids) - set(self.envelope.source_records):
            raise CognitiveKernelContractError(
                "conflict envelope must bind every member record"
            )
        if set(self.evidence_relation_ids) - set(
            self.envelope.source_records
        ):
            raise CognitiveKernelContractError(
                "conflict envelope must bind every evidence relation"
            )
        require_sha256(self.conflict_sha256, "conflict_sha256")
        if canonical_sha256(self.material_record()) != self.conflict_sha256:
            raise CognitiveKernelContractError("conflict digest mismatch")


@dataclass(frozen=True)
class AdjudicationRecord:
    """Inspectable decision over one candidate without conflating confidence."""

    envelope: MemoryUnitEnvelope
    adjudication_id: str
    candidate_id: str
    claim_id: str
    authority_class: str
    authority_actor_id: str
    policy_profile: str
    rule_id: str
    rule_version: str
    evidence_relation_ids: tuple[str, ...]
    alternatives: tuple[str, ...]
    confidence: float | None
    outcome: str
    execution_mode: str
    canonical_effect: bool
    conflict_record_id: str | None
    rationale_codes: tuple[str, ...]
    rollback_reference: str
    adjudication_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        adjudication_id: object,
        candidate_id: object,
        claim_id: object,
        authority_class: object,
        authority_actor_id: object,
        policy_profile: object,
        rule_id: object,
        rule_version: object,
        evidence_relation_ids: Iterable[object] = (),
        alternatives: Iterable[object] = (),
        confidence: float | None,
        outcome: object,
        execution_mode: object,
        canonical_effect: bool,
        conflict_record_id: object | None,
        rationale_codes: Iterable[object],
        rollback_reference: object,
    ) -> "AdjudicationRecord":
        normalized_id = require_identifier(
            adjudication_id, "adjudication_id"
        )
        draft = cls(
            envelope=envelope,
            adjudication_id=normalized_id,
            candidate_id=require_identifier(candidate_id, "candidate_id"),
            claim_id=require_identifier(claim_id, "claim_id"),
            authority_class=require_identifier(
                authority_class, "authority_class"
            ),
            authority_actor_id=require_identifier(
                authority_actor_id, "authority_actor_id"
            ),
            policy_profile=require_identifier(
                policy_profile, "policy_profile"
            ),
            rule_id=require_identifier(rule_id, "rule_id"),
            rule_version=require_identifier(rule_version, "rule_version"),
            evidence_relation_ids=_sorted_identifiers(
                evidence_relation_ids, "evidence_relation_ids"
            ),
            alternatives=_sorted_identifiers(
                alternatives, "alternatives"
            ),
            confidence=(
                require_confidence(confidence, "confidence")
                if confidence is not None
                else None
            ),
            outcome=require_identifier(outcome, "outcome"),
            execution_mode=require_identifier(
                execution_mode, "execution_mode"
            ),
            canonical_effect=bool(canonical_effect),
            conflict_record_id=_optional_identifier(
                conflict_record_id, "conflict_record_id"
            ),
            rationale_codes=_sorted_identifiers(
                rationale_codes, "rationale_codes", minimum=1
            ),
            rollback_reference=require_identifier(
                rollback_reference, "rollback_reference"
            ),
            adjudication_sha256="0" * 64,
        )
        result = cls(
            **{
                **draft.__dict__,
                "adjudication_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": ADJUDICATION_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "adjudication_id": self.adjudication_id,
            "candidate_id": self.candidate_id,
            "claim_id": self.claim_id,
            "authority_class": self.authority_class,
            "authority_actor_id": self.authority_actor_id,
            "policy_profile": self.policy_profile,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "evidence_relation_ids": list(self.evidence_relation_ids),
            "alternatives": list(self.alternatives),
            "confidence": self.confidence,
            "outcome": self.outcome,
            "execution_mode": self.execution_mode,
            "canonical_effect": self.canonical_effect,
            "conflict_record_id": self.conflict_record_id,
            "rationale_codes": list(self.rationale_codes),
            "rollback_reference": self.rollback_reference,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["adjudication_sha256"] = self.adjudication_sha256
        return record

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_id=self.adjudication_id,
            record_type="adjudication_record",
            authority_roles=frozenset(
                {"evaluation_artifact", "claim_authority"}
            ),
        )
        if self.authority_class not in ADJUDICATION_AUTHORITY_CLASSES:
            raise CognitiveKernelContractError(
                "authority_class is not ratified"
            )
        if self.outcome not in ADJUDICATION_OUTCOMES:
            raise CognitiveKernelContractError("outcome is not ratified")
        if self.execution_mode not in ADJUDICATION_EXECUTION_MODES:
            raise CognitiveKernelContractError(
                "execution_mode is not ratified"
            )
        if self.execution_mode == "shadow" and self.canonical_effect:
            raise CognitiveKernelContractError(
                "shadow adjudication cannot have canonical effect"
            )
        required_sources = {
            self.candidate_id,
            self.claim_id,
            *self.evidence_relation_ids,
        }
        if self.conflict_record_id is not None:
            required_sources.add(self.conflict_record_id)
        if required_sources - set(self.envelope.source_records):
            raise CognitiveKernelContractError(
                "adjudication envelope is missing bound source records"
            )
        require_sha256(
            self.adjudication_sha256, "adjudication_sha256"
        )
        if (
            canonical_sha256(self.material_record())
            != self.adjudication_sha256
        ):
            raise CognitiveKernelContractError(
                "adjudication digest mismatch"
            )

    def assert_adjudicates(self, candidate: ClaimCandidate) -> None:
        self.validate()
        candidate.validate()
        if self.candidate_id != candidate.candidate_id:
            raise CognitiveKernelContractError(
                "adjudication candidate_id differs from candidate"
            )
        if self.claim_id != candidate.identity.claim_id:
            raise CognitiveKernelContractError(
                "adjudication claim_id differs from candidate identity"
            )
        if not _same_scope(self.envelope.scope, candidate.envelope.scope):
            raise CognitiveKernelContractError(
                "adjudication crosses product-host-encryption scope"
            )
        if (
            self.envelope.authority_namespace_id
            != candidate.envelope.authority_namespace_id
        ):
            raise CognitiveKernelContractError(
                "adjudication crosses authority namespace"
            )
