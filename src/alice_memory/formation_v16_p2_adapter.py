"""A limited MFM 1.6 output bridge into the existing P2 candidate gate.

Only an owner profile claim grounded in currently registered, opened text can
be represented by P2. Rich 1.6 semantics are rejected, never flattened into a
P2 category or silently dropped. The returned receipt binds the exact input,
output and registration snapshot; P2 does not yet persist this full receipt.
Neither adaptation nor staging assesses or promotes a candidate.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Callable

from cognitive_kernel.canonical import (
    canonical_sha256, require_identifier, require_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import FormationEvidenceRef
from cognitive_kernel.formation_semantics_v16 import (
    FormationContextV16, SENSITIVITY_ORDER, bundle_v16_from_output,
    validate_formation_grounding_v16,
)

from .formation import (
    MemoryCandidateCreateRequest, MemoryCandidateRecord,
    MemoryCandidateWriteAuthorization, propose_memory_candidate,
)
from .service import _normalize_timestamp
from .sources import MemorySourceSpec, validate_memory_source


ADAPTER_VERSION = "mfm-v16-p2-owner-profile-v1"
_CANDIDATE_NAMESPACE = uuid.UUID("c74d680c-5205-4bba-88aa-c92a850b0e81")


class FormationP2AdapterError(ValueError):
    """The output cannot safely be represented by the current P2 gate."""


@dataclass(frozen=True)
class P2FormationGateBinding:
    """Authority-owned binding for one Alice host's private P2 store.

    The caller must get this binding from the store's trusted configuration,
    not from the model context. An artifact digest is a pin, not proof that
    the weights were trained, approved, or independently evaluated.
    """

    scope: ProductHostScope
    authority_namespace_id: str
    owner_subject_ref: str
    approved_artifact_sha256: str

    def validate(self) -> None:
        self.scope.validate()
        if self.scope.product_id != "alice":
            raise FormationP2AdapterError("P2 adapter requires an Alice host store")
        for field in ("authority_namespace_id", "owner_subject_ref"):
            value = getattr(self, field)
            if require_identifier(value, field) != value:
                raise FormationP2AdapterError(f"{field} must be canonical")
        if require_sha256(self.approved_artifact_sha256, "approved_artifact_sha256") != self.approved_artifact_sha256:
            raise FormationP2AdapterError("artifact digest must be canonical")


@dataclass(frozen=True)
class CurrentFormationSource:
    """Live authority lookup, including its current policy and P2 locator."""

    evidence: FormationEvidenceRef
    sensitivity: str
    source: MemorySourceSpec
    active: bool
    candidate_write_allowed: bool


@dataclass(frozen=True)
class FormationP2Receipt:
    """Metadata-safe, deterministic binding for external authorized custody."""

    candidate_id: str
    binding_sha256: str
    context_v16_sha256: str
    bundle_sha256: str
    proposal_sha256: str
    content_sha256: str
    model_artifact_sha256: str
    inference_run_id: str
    evidence_refs: tuple[str, ...]
    source_sha256: tuple[str, ...]
    data_classification: str
    adapter_version: str = ADAPTER_VERSION


@dataclass(frozen=True)
class PreparedP2FormationCandidate:
    request: MemoryCandidateCreateRequest
    receipt: FormationP2Receipt


@dataclass(frozen=True)
class StagedP2FormationCandidate:
    candidate: MemoryCandidateRecord
    receipt: FormationP2Receipt


def _current_source(
    ref: FormationEvidenceRef,
    resolve_source: Callable[[str], CurrentFormationSource | None],
) -> CurrentFormationSource:
    registration = resolve_source(ref.ref_id)
    if not isinstance(registration, CurrentFormationSource):
        raise FormationP2AdapterError("cited source is not currently registered")
    if registration.active is not True or registration.candidate_write_allowed is not True:
        raise FormationP2AdapterError("cited source is revoked or disallows candidate writes")
    if registration.evidence != ref:
        raise FormationP2AdapterError("cited source identity or metadata changed")
    if registration.sensitivity not in SENSITIVITY_ORDER:
        raise FormationP2AdapterError("cited source has no supported current sensitivity")
    source = registration.source
    validate_memory_source(source)
    if (source.source_type != "rayan_direct_statement" or
            source.source_ref != ref.ref_id or
            source.support_relation != "derived_from" or
            source.source_content_sha256 != ref.content_digest or
            source.source_text_sha256 != ref.content_digest or
            source.source_date != ref.observed_at):
        raise FormationP2AdapterError("P2 source locator, hash, date or relation differs")
    return registration


def prepare_p2_candidate_v16(
    *,
    context: FormationContextV16,
    output: object,
    opened_sources: tuple[tuple[str, bytes], ...],
    binding: P2FormationGateBinding,
    resolve_source: Callable[[str], CurrentFormationSource | None],
    artifact_sha256: str,
    inference_run_id: str,
    proposed_at: str,
) -> PreparedP2FormationCandidate:
    """Validate a raw 1.6 output against exact bytes and live source policy.

    The resolver must read the authority's current source registry. A caller
    must retain the receipt in authorized custody and recheck source policy at
    promotion; P2's own gate does not yet know the 1.6 registry or receipt.
    """
    binding.validate()
    if context.base.scope != binding.scope or context.base.authority_namespace_id != binding.authority_namespace_id:
        raise FormationP2AdapterError("formation context targets a different host or authority")
    if artifact_sha256 != binding.approved_artifact_sha256:
        raise FormationP2AdapterError("model artifact is not pinned by the P2 authority")
    bundle = bundle_v16_from_output(
        context, output, artifact_sha256=artifact_sha256,
        inference_run_id=inference_run_id,
    )
    validate_formation_grounding_v16(context, bundle, opened_sources)
    if len(bundle.proposals) != 1 or len(bundle.base.dispositions) != 1:
        raise FormationP2AdapterError("P2 subset requires exactly one proposal and disposition")
    proposal = bundle.proposals[0]
    base = proposal.base
    disposition = bundle.base.dispositions[0]
    if (disposition.action != "propose" or
            disposition.scope_ref != base.disposition_scope_ref or
            disposition.evidence_refs != base.evidence_refs or
            disposition.target_refs):
        raise FormationP2AdapterError("P2 subset requires an exact propose disposition")
    if (base.kind != "claim" or base.domain != "host" or
            base.subject_ref != binding.owner_subject_ref or
            base.epistemic_status != "owner_statement"):
        raise FormationP2AdapterError("P2 subset supports only an owner's host profile claim")
    if (proposal.episode is not None or proposal.relationship_counterpart_ref is not None or
            proposal.mission_target_refs or proposal.workspace_target_refs or
            base.contradicts or base.target_refs or base.uncertainty_ref or
            base.temporal_granularity != "instant"):
        raise FormationP2AdapterError("P2 cannot represent this v1.6 semantic dimension")
    if base.confidence is None:
        raise FormationP2AdapterError("P2 requires an explicit, uncalibrated model confidence")

    evidence = {ref.ref_id: ref for ref in context.base.evidence}
    floor = {item.ref_id: item.minimum for item in context.source_sensitivities}
    registrations = []
    for ref_id in base.evidence_refs:
        ref = evidence[ref_id]
        if (ref.role != "authenticated_owner_statement" or ref.modality != "text" or
                ref.subject_ref != binding.owner_subject_ref or
                ref.speaker_ref != binding.owner_subject_ref):
            raise FormationP2AdapterError("P2 owner claim requires authenticated owner text")
        registrations.append(_current_source(ref, resolve_source))
    sensitivity_index = max(
        *(SENSITIVITY_ORDER.index(floor[ref_id]) for ref_id in base.evidence_refs),
        *(SENSITIVITY_ORDER.index(item.sensitivity) for item in registrations),
        *([SENSITIVITY_ORDER.index(proposal.sensitivity_hint)]
          if proposal.sensitivity_hint is not None else []),
    )
    if sensitivity_index >= SENSITIVITY_ORDER.index("highly_sensitive"):
        raise FormationP2AdapterError("P2 candidate storage cannot hold highly sensitive content")
    classification = SENSITIVITY_ORDER[sensitivity_index].upper()
    normalized_at = _normalize_timestamp(proposed_at, field_name="proposed_at")
    source_specs = tuple(item.source for item in registrations)
    binding_record = {
        "adapter_version": ADAPTER_VERSION,
        "scope": binding.scope.metadata_record(),
        "authority_namespace_id": binding.authority_namespace_id,
        "owner_subject_ref": binding.owner_subject_ref,
        "context_v16_sha256": context.content_digest(),
        "bundle_sha256": bundle.content_digest(),
        "proposal_sha256": canonical_sha256(proposal.record()),
        "model_artifact_sha256": artifact_sha256,
        "inference_run_id": inference_run_id,
        "recorded_at": normalized_at,
        "content_sha256": hashlib.sha256(base.value_text.encode("utf-8")).hexdigest(),
        "classification": classification,
        "sources": [{
            "ref_id": ref_id,
            "evidence_sha256": evidence[ref_id].content_digest,
            "evidence_metadata_sha256": canonical_sha256(evidence[ref_id].metadata_record()),
            "current_sensitivity": registration.sensitivity,
            "source_type": registration.source.source_type,
            "source_ref": registration.source.source_ref,
            "source_content_sha256": registration.source.source_content_sha256,
            "source_text_sha256": registration.source.source_text_sha256,
        } for ref_id, registration in zip(base.evidence_refs, registrations)],
    }
    digest = canonical_sha256(binding_record)
    candidate_id = str(uuid.uuid5(_CANDIDATE_NAMESPACE, digest))
    receipt = FormationP2Receipt(
        candidate_id=candidate_id, binding_sha256=digest,
        context_v16_sha256=context.content_digest(),
        bundle_sha256=bundle.content_digest(),
        proposal_sha256=canonical_sha256(proposal.record()),
        content_sha256=hashlib.sha256(base.value_text.encode("utf-8")).hexdigest(),
        model_artifact_sha256=artifact_sha256,
        inference_run_id=inference_run_id,
        evidence_refs=base.evidence_refs,
        source_sha256=tuple(evidence[ref].content_digest for ref in base.evidence_refs),
        data_classification=classification,
    )
    request = MemoryCandidateCreateRequest(
        candidate_id=candidate_id, content=base.value_text,
        category="profile", knowledge_status="alice_inference",
        confidence=base.confidence, data_classification=classification,
        recorded_at=normalized_at, sources=source_specs, origin="model_proposed",
        valid_from=base.valid_from, valid_to=base.valid_to,
        rayan_confirmed=False, policy_version=ADAPTER_VERSION,
        model="mfm-formation-v1.6", model_version=artifact_sha256,
        prompt_version="mfm-formation-output-v1.6", run_id=inference_run_id,
    )
    return PreparedP2FormationCandidate(request, receipt)


def stage_p2_candidate_v16(
    connection: sqlite3.Connection,
    *,
    authorization: MemoryCandidateWriteAuthorization,
    context: FormationContextV16,
    output: object,
    opened_sources: tuple[tuple[str, bytes], ...],
    binding: P2FormationGateBinding,
    resolve_source: Callable[[str], CurrentFormationSource | None],
    artifact_sha256: str,
    inference_run_id: str,
    proposed_at: str,
) -> StagedP2FormationCandidate:
    """Stage via P2's authorized candidate write; never assess or promote."""
    prepared = prepare_p2_candidate_v16(
        context=context, output=output, opened_sources=opened_sources,
        binding=binding, resolve_source=resolve_source,
        artifact_sha256=artifact_sha256, inference_run_id=inference_run_id,
        proposed_at=proposed_at,
    )
    candidate = propose_memory_candidate(
        connection, request=prepared.request, authorization=authorization,
        proposed_at=proposed_at,
    )
    return StagedP2FormationCandidate(candidate, prepared.receipt)
