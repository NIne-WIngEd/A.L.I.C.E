"""Frozen-gold assessment of non-authoritative formation outputs.

This assesses learned output, not backend qualification or memory-gate decisions.
It fails closed on identity/provenance errors instead of hiding them in averages.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import CognitiveKernelContractError, normalize_timestamp, require_identifier
from .formation_contracts import (
    FormationContextPacket,
    FormationProposal,
    MemoryProposalBundle,
    validate_formation_binding,
)


def _label(p: FormationProposal) -> tuple[str, str, str, str, str]:
    return (p.kind, p.domain, p.subject_ref, p.value_ref, p.epistemic_status)


def _identity(p: FormationProposal) -> tuple[object, ...]:
    return (
        *_label(p), tuple(sorted(p.evidence_refs)),
        normalize_timestamp(p.valid_from) if p.valid_from else None,
        normalize_timestamp(p.valid_to) if p.valid_to else None,
        p.uncertainty_ref, tuple(sorted(p.contradicts)),
    )


@dataclass(frozen=True)
class FormationGoldCase:
    case_id: str
    context: FormationContextPacket
    expected: tuple[FormationProposal, ...]
    critical_forbidden: tuple[tuple[str, str, str, str, str], ...] = ()

    def validate(self) -> None:
        require_identifier(self.case_id, "case_id")
        self.context.validate()
        seen: set[tuple[object, ...]] = set()
        refs = {r.ref_id for r in self.context.evidence}
        for proposal in self.expected:
            proposal.validate()
            if not set(proposal.evidence_refs).issubset(refs):
                raise CognitiveKernelContractError("gold cites evidence absent from context")
            key = _identity(proposal)
            if key in seen:
                raise CognitiveKernelContractError("duplicate gold semantic proposal")
            seen.add(key)
        for key in self.critical_forbidden:
            if len(key) != 5 or key in {item[:5] for item in seen}:
                raise CognitiveKernelContractError("invalid critical forbidden gold label")


@dataclass(frozen=True)
class FormationAssessment:
    case_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    critical_failures: tuple[str, ...]

    @property
    def passes_critical_gate(self) -> bool:
        return not self.critical_failures


def assess_formation(
    gold: FormationGoldCase,
    output: MemoryProposalBundle,
) -> FormationAssessment:
    gold.validate()
    if output.scope != gold.context.scope or output.authority_namespace_id != gold.context.authority_namespace_id:
        raise CognitiveKernelContractError("assessment output has foreign host scope")
    if output.context_digest != gold.context.content_digest():
        raise CognitiveKernelContractError("assessment output has wrong context digest")
    validate_formation_binding(gold.context, output)
    expected = {_identity(p) for p in gold.expected}
    predicted = {_identity(p) for p in output.proposals}
    if len(predicted) != len(output.proposals):
        raise CognitiveKernelContractError("duplicate semantic formation output")
    predicted_labels = {_label(p) for p in output.proposals}
    critical = tuple(
        "/".join(key) for key in gold.critical_forbidden if key in predicted_labels
    )
    return FormationAssessment(
        case_id=gold.case_id,
        true_positives=len(expected & predicted),
        false_positives=len(predicted - expected),
        false_negatives=len(expected - predicted),
        critical_failures=critical,
    )
