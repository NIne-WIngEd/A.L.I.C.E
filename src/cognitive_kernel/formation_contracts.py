"""Backend-neutral output contract for a learned Memory Formation Model.

These records are proposals. Neither construction nor validation writes a memory,
confirms an assertion, or grants a model permission to choose an authority store.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_confidence,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import ProductHostScope

FORMATION_SCHEMA_VERSION = "1.1.0"
PROPOSAL_KINDS = frozenset({
    "claim", "preference", "relationship", "episode", "goal", "mission",
    "host_observation", "source_person_evidence", "self_observation",
    "correction_request", "deletion_request", "revocation_request",
    "contradiction", "uncertainty", "decision_rationale", "outcome",
    "behavior_pattern", "relationship_norm", "procedural_skill",
    "metacognitive_signal",
})
MEMORY_DOMAINS = frozenset({
    "source_person", "host", "relationship", "self", "mission", "world",
})
EPISTEMIC_STATUSES = frozenset({
    "owner_statement", "source_person_attestation", "observation",
    "external_claim", "inference", "prediction", "reconstruction",
    "hypothetical", "uncertain",
})
EVIDENCE_ROLES = frozenset({
    "authenticated_owner_statement", "owner_attested_source_person",
    "tool_or_action_observation", "assistant_self_event", "outside_source",
    "derived_inference", "generated_reconstruction", "hypothetical_training",
    "authoritative_claim", "historical_experience", "mission_state",
})
EVIDENCE_MODALITIES = frozenset({
    "text", "image", "audio", "video", "code", "structured",
    "sensor", "action", "multimodal",
})


def _canonical_id(value: str, field: str) -> None:
    if require_identifier(value, field) != value:
        raise CognitiveKernelContractError(f"{field} must be canonical")


@dataclass(frozen=True)
class FormationEvidenceRef:
    """Vetted context metadata bound to evidence bytes held in authorized custody."""

    ref_id: str
    scope: ProductHostScope
    authority_namespace_id: str
    content_digest: str
    role: str
    modality: str
    subject_ref: str | None = None

    def validate(self) -> None:
        self.scope.validate()
        _canonical_id(self.ref_id, "ref_id")
        _canonical_id(self.authority_namespace_id, "authority_namespace_id")
        if require_sha256(self.content_digest, "content_digest") != self.content_digest:
            raise CognitiveKernelContractError("content_digest must be canonical")
        if self.role not in EVIDENCE_ROLES:
            raise CognitiveKernelContractError("unsupported evidence role")
        if self.modality not in EVIDENCE_MODALITIES:
            raise CognitiveKernelContractError("unsupported evidence modality")
        if self.subject_ref is not None:
            _canonical_id(self.subject_ref, "subject_ref")

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "ref_id": self.ref_id, "scope": self.scope.metadata_record(),
            "authority_namespace_id": self.authority_namespace_id,
            "content_digest": self.content_digest, "role": self.role,
            "modality": self.modality, "subject_ref": self.subject_ref,
        }


@dataclass(frozen=True)
class FormationContextPacket:
    """A model input manifest constructed from registered, scoped evidence."""

    scope: ProductHostScope
    authority_namespace_id: str
    experience_refs: tuple[str, ...]
    evidence: tuple[FormationEvidenceRef, ...]
    schema_version: str = FORMATION_SCHEMA_VERSION

    def validate(self) -> None:
        self.scope.validate()
        if self.schema_version != FORMATION_SCHEMA_VERSION:
            raise CognitiveKernelContractError("unsupported formation schema version")
        _canonical_id(self.authority_namespace_id, "authority_namespace_id")
        if not self.experience_refs:
            raise CognitiveKernelContractError("context needs an experience reference")
        if normalize_identifier_sequence(self.experience_refs, "experience_refs") != self.experience_refs:
            raise CognitiveKernelContractError("experience_refs must be canonical")
        ids: set[str] = set()
        for ref in self.evidence:
            ref.validate()
            if ref.scope != self.scope or ref.authority_namespace_id != self.authority_namespace_id:
                raise CognitiveKernelContractError("context contains foreign-scope evidence")
            if ref.ref_id in ids:
                raise CognitiveKernelContractError("duplicate context evidence reference")
            ids.add(ref.ref_id)
        if not set(self.experience_refs).issubset(ids):
            raise CognitiveKernelContractError("experience is missing from context evidence")

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "scope": self.scope.metadata_record(),
            "authority_namespace_id": self.authority_namespace_id,
            "experience_refs": list(self.experience_refs),
            "evidence": [ref.metadata_record() for ref in self.evidence],
            "schema_version": self.schema_version,
        }

    def content_digest(self) -> str:
        return canonical_sha256(self.metadata_record())


@dataclass(frozen=True)
class FormationProposal:
    """One interpretation of evidence, with no canonical authority."""

    proposal_id: str
    kind: str
    domain: str
    subject_ref: str
    value_ref: str
    evidence_refs: tuple[str, ...]
    epistemic_status: str = "uncertain"
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float | None = None
    uncertainty_ref: str | None = None
    contradicts: tuple[str, ...] = ()

    def validate(self) -> None:
        _canonical_id(self.proposal_id, "proposal_id")
        if self.kind not in PROPOSAL_KINDS:
            raise CognitiveKernelContractError("unsupported formation proposal kind")
        if self.domain not in MEMORY_DOMAINS:
            raise CognitiveKernelContractError("unsupported memory domain")
        if self.epistemic_status not in EPISTEMIC_STATUSES:
            raise CognitiveKernelContractError("unsupported epistemic status")
        _canonical_id(self.subject_ref, "subject_ref")
        _canonical_id(self.value_ref, "value_ref")
        if not self.evidence_refs:
            raise CognitiveKernelContractError("a proposal needs evidence references")
        if normalize_identifier_sequence(self.evidence_refs, "evidence_refs") != self.evidence_refs:
            raise CognitiveKernelContractError("evidence_refs must be canonical")
        if normalize_identifier_sequence(self.contradicts, "contradicts") != self.contradicts:
            raise CognitiveKernelContractError("contradicts must be canonical")
        if self.valid_from is not None:
            normalize_timestamp(self.valid_from, "valid_from")
        if self.valid_to is not None:
            normalize_timestamp(self.valid_to, "valid_to")
        if self.valid_from and self.valid_to:
            if normalize_timestamp(self.valid_to) < normalize_timestamp(self.valid_from):
                raise CognitiveKernelContractError("valid_to precedes valid_from")
        require_confidence(self.confidence)
        if self.uncertainty_ref is not None:
            _canonical_id(self.uncertainty_ref, "uncertainty_ref")

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "proposal_id": self.proposal_id, "kind": self.kind,
            "domain": self.domain, "subject_ref": self.subject_ref,
            "value_ref": self.value_ref, "evidence_refs": list(self.evidence_refs),
            "epistemic_status": self.epistemic_status,
            "valid_from": normalize_timestamp(self.valid_from) if self.valid_from else None,
            "valid_to": normalize_timestamp(self.valid_to) if self.valid_to else None,
            "confidence": self.confidence, "uncertainty_ref": self.uncertainty_ref,
            "contradicts": list(self.contradicts),
        }


@dataclass(frozen=True)
class MemoryProposalBundle:
    """A scoped, traceable model output for an independent authority gate."""

    scope: ProductHostScope
    authority_namespace_id: str
    bundle_id: str
    experience_refs: tuple[str, ...]
    context_digest: str
    model_artifact_digest: str
    inference_run_id: str
    proposals: tuple[FormationProposal, ...]
    schema_version: str = FORMATION_SCHEMA_VERSION

    def validate(self) -> None:
        self.scope.validate()
        require_schema_version(self.schema_version)
        if self.schema_version != FORMATION_SCHEMA_VERSION:
            raise CognitiveKernelContractError("unsupported formation schema version")
        _canonical_id(self.authority_namespace_id, "authority_namespace_id")
        _canonical_id(self.bundle_id, "bundle_id")
        if not self.experience_refs:
            raise CognitiveKernelContractError("bundle needs an experience reference")
        if normalize_identifier_sequence(self.experience_refs, "experience_refs") != self.experience_refs:
            raise CognitiveKernelContractError("experience_refs must be canonical")
        if require_sha256(self.context_digest, "context_digest") != self.context_digest:
            raise CognitiveKernelContractError("context_digest must be canonical")
        if require_sha256(self.model_artifact_digest, "model_artifact_digest") != self.model_artifact_digest:
            raise CognitiveKernelContractError("model_artifact_digest must be canonical")
        _canonical_id(self.inference_run_id, "inference_run_id")
        ids: set[str] = set()
        for proposal in self.proposals:
            proposal.validate()
            if proposal.proposal_id in ids:
                raise CognitiveKernelContractError("duplicate proposal_id")
            ids.add(proposal.proposal_id)

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "scope": self.scope.metadata_record(),
            "authority_namespace_id": self.authority_namespace_id,
            "bundle_id": self.bundle_id,
            "experience_refs": list(self.experience_refs),
            "context_digest": self.context_digest,
            "model_artifact_digest": self.model_artifact_digest,
            "inference_run_id": self.inference_run_id,
            "proposals": [proposal.record() for proposal in self.proposals],
            "schema_version": self.schema_version,
        }

    def content_digest(self) -> str:
        return canonical_sha256(self.metadata_record())


def validate_formation_binding(
    context: FormationContextPacket,
    bundle: MemoryProposalBundle,
) -> None:
    """Check provenance claims against the immutable model input manifest.

    Successful binding does not make any proposal true or authorize a write.
    The authority gate must still check actors, evidence, privacy, conflicts,
    deletion and product policies against its own registered sources.
    """

    context.validate()
    bundle.validate()
    if context.scope != bundle.scope or context.authority_namespace_id != bundle.authority_namespace_id:
        raise CognitiveKernelContractError("formation scope does not match context")
    if context.experience_refs != bundle.experience_refs:
        raise CognitiveKernelContractError("formation experience does not match context")
    if context.content_digest() != bundle.context_digest:
        raise CognitiveKernelContractError("formation context digest does not match")
    evidence = {ref.ref_id: ref for ref in context.evidence}
    compatible_roles = {
        "owner_statement": "authenticated_owner_statement",
        "source_person_attestation": "owner_attested_source_person",
        "external_claim": "outside_source",
        "hypothetical": "hypothetical_training",
        "reconstruction": "generated_reconstruction",
    }
    for proposal in bundle.proposals:
        if not set(proposal.evidence_refs).issubset(evidence):
            raise CognitiveKernelContractError("proposal cites evidence absent from context")
        required = compatible_roles.get(proposal.epistemic_status)
        if required and not all(evidence[ref].role == required for ref in proposal.evidence_refs):
            raise CognitiveKernelContractError("proposal epistemic status lacks matching evidence")
