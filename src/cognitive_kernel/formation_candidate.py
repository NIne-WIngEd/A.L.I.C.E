"""Candidate boundary and source-separated evaluation for a learned MFM.

This runs externally trained weights through frozen formation gold. It neither
implements a rule-based substitute nor treats schema tests as model ability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canonical import CognitiveKernelContractError, require_identifier, require_sha256
from .formation_contracts import FormationContextPacket, MemoryProposalBundle, validate_formation_grounding
from .formation_evaluation import FormationAssessment, assess_formation
from .formation_gold import CompiledFormationCase


class LearnedFormationCandidate(Protocol):
    artifact_sha256: str

    def infer(
        self, *, context: FormationContextPacket,
        opened_sources: tuple[tuple[str, bytes], ...],
    ) -> MemoryProposalBundle: ...


class GroundingReviewCallback(Protocol):
    """Secondary check; distinct artifacts do not authenticate independence."""

    reviewer_id: str
    artifact_sha256: str

    def unsupported_proposal_ids(
        self, *, context: FormationContextPacket,
        opened_sources: tuple[tuple[str, bytes], ...],
        output: MemoryProposalBundle,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class CandidateCaseResult:
    case_id: str
    split: str
    assessment: FormationAssessment
    output_digest: str
    reviewer_flagged_unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateEvaluation:
    artifact_sha256: str
    cases: tuple[CandidateCaseResult, ...]
    review_callback_used: bool = False

    @property
    def critical_failures(self) -> tuple[str, ...]:
        return tuple(f"{case.case_id}:{failure}" for case in self.cases
                     for failure in (*case.assessment.critical_failures,
                                     *(f"unsupported:{item}" for item in case.reviewer_flagged_unsupported)))

    @property
    def totals(self) -> dict[str, int]:
        return {
            "true_positives": sum(c.assessment.true_positives for c in self.cases),
            "false_positives": sum(c.assessment.false_positives for c in self.cases),
            "false_negatives": sum(c.assessment.false_negatives for c in self.cases),
            "critical_failures": len(self.critical_failures),
        }


def training_examples(cases: tuple[CompiledFormationCase, ...]) -> tuple[CompiledFormationCase, ...]:
    """Reject the old diagnostic fixture path as an optimizer data handoff.

    A real trainer must receive byte paths from CorpusAdmission.audit_handoff
    after authenticated rights/review checks, never this public fixture loader.
    """
    raise CognitiveKernelContractError(
        "public formation diagnostic cases are not admitted training examples")


def evaluate_candidate(
    candidate: LearnedFormationCandidate,
    cases: tuple[CompiledFormationCase, ...], *, split: str,
    grounding_reviewer: GroundingReviewCallback | None = None,
) -> CandidateEvaluation:
    digest = require_sha256(candidate.artifact_sha256, "artifact_sha256")
    if not cases or split not in {"development", "challenge"}:
        raise CognitiveKernelContractError("candidate evaluation requires development or challenge gold")
    selected = tuple(case for case in cases if case.split == split)
    if not selected:
        raise CognitiveKernelContractError("requested gold split is empty")
    results = []
    if grounding_reviewer is not None:
        require_identifier(grounding_reviewer.reviewer_id, "reviewer_id")
        require_sha256(grounding_reviewer.artifact_sha256, "reviewer_artifact_sha256")
        if grounding_reviewer.artifact_sha256 == digest:
            raise CognitiveKernelContractError("grounding reviewer cannot be the candidate artifact")
    for case in selected:
        opened = tuple((ref_id, text.encode("utf-8")) for ref_id, text in case.texts)
        output = candidate.infer(context=case.gold.context, opened_sources=opened)
        if not isinstance(output, MemoryProposalBundle):
            raise CognitiveKernelContractError("v1.5 candidate evaluator only accepts v1.5 output")
        if output.model_artifact_digest != digest:
            raise CognitiveKernelContractError("candidate output cites different model artifact")
        validate_formation_grounding(case.gold.context, output, opened)
        assessment = assess_formation(case.gold, output)
        unsupported: tuple[str, ...] = ()
        if grounding_reviewer is not None:
            unsupported = grounding_reviewer.unsupported_proposal_ids(
                context=case.gold.context, opened_sources=opened, output=output)
            if (not isinstance(unsupported, tuple) or len(set(unsupported)) != len(unsupported)
                    or not set(unsupported).issubset({p.proposal_id for p in output.proposals})):
                raise CognitiveKernelContractError("grounding reviewer returned invalid proposal IDs")
        results.append(CandidateCaseResult(case.gold.case_id, split, assessment,
                                           output.content_digest(), unsupported))
    return CandidateEvaluation(digest, tuple(results), grounding_reviewer is not None)
