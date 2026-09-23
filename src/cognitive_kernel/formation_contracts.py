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

FORMATION_SCHEMA_VERSION = "1.0.0"
PROPOSAL_KINDS = frozenset({
    "claim", "preference", "relationship", "episode", "goal", "mission",
    "host_observation", "source_person_evidence", "self_observation",
    "correction_request", "deletion_request", "revocation_request",
    "contradiction", "uncertainty",
})
MEMORY_DOMAINS = frozenset({
    "source_person", "host", "relationship", "self", "mission", "world",
})


@dataclass(frozen=True)
class FormationProposal:
    """One interpretation of evidence, with no canonical authority."""

    proposal_id: str
    kind: str
    domain: str
    subject_ref: str
    value_ref: str
    evidence_refs: tuple[str, ...]
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float | None = None
    uncertainty_ref: str | None = None
    contradicts: tuple[str, ...] = ()

    def validate(self) -> None:
        require_identifier(self.proposal_id, "proposal_id")
        if self.kind not in PROPOSAL_KINDS:
            raise CognitiveKernelContractError("unsupported formation proposal kind")
        if self.domain not in MEMORY_DOMAINS:
            raise CognitiveKernelContractError("unsupported memory domain")
        require_identifier(self.subject_ref, "subject_ref")
        require_identifier(self.value_ref, "value_ref")
        if not self.evidence_refs:
            raise CognitiveKernelContractError("a proposal needs evidence references")
        normalize_identifier_sequence(self.evidence_refs, "evidence_refs")
        normalize_identifier_sequence(self.contradicts, "contradicts")
        if self.valid_from is not None:
            normalize_timestamp(self.valid_from, "valid_from")
        if self.valid_to is not None:
            normalize_timestamp(self.valid_to, "valid_to")
        if self.valid_from and self.valid_to:
            if normalize_timestamp(self.valid_to) < normalize_timestamp(self.valid_from):
                raise CognitiveKernelContractError("valid_to precedes valid_from")
        require_confidence(self.confidence)
        if self.uncertainty_ref is not None:
            require_identifier(self.uncertainty_ref, "uncertainty_ref")

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "proposal_id": self.proposal_id, "kind": self.kind,
            "domain": self.domain, "subject_ref": self.subject_ref,
            "value_ref": self.value_ref, "evidence_refs": list(self.evidence_refs),
            "valid_from": self.valid_from, "valid_to": self.valid_to,
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
        require_identifier(self.authority_namespace_id, "authority_namespace_id")
        require_identifier(self.bundle_id, "bundle_id")
        if not self.experience_refs:
            raise CognitiveKernelContractError("bundle needs an experience reference")
        normalize_identifier_sequence(self.experience_refs, "experience_refs")
        require_sha256(self.context_digest, "context_digest")
        require_sha256(self.model_artifact_digest, "model_artifact_digest")
        require_identifier(self.inference_run_id, "inference_run_id")
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
