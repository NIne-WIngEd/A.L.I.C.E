"""Candidate boundary and source-separated evaluation for a learned MFM.

This runs externally trained weights through frozen formation gold. It neither
implements a rule-based substitute nor treats schema tests as model ability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canonical import CognitiveKernelContractError, require_sha256
from .formation_contracts import FormationContextPacket, MemoryProposalBundle
from .formation_evaluation import FormationAssessment, assess_formation
from .formation_gold import CompiledFormationCase


class LearnedFormationCandidate(Protocol):
    artifact_sha256: str

    def infer(
        self, *, context: FormationContextPacket,
        opened_sources: tuple[tuple[str, bytes], ...],
    ) -> MemoryProposalBundle: ...


@dataclass(frozen=True)
class CandidateCaseResult:
    case_id: str
    split: str
    assessment: FormationAssessment
    output_digest: str


@dataclass(frozen=True)
class CandidateEvaluation:
    artifact_sha256: str
    cases: tuple[CandidateCaseResult, ...]

    @property
    def critical_failures(self) -> tuple[str, ...]:
        return tuple(f"{case.case_id}:{failure}"
                     for case in self.cases for failure in case.assessment.critical_failures)

    @property
    def totals(self) -> dict[str, int]:
        return {
            "true_positives": sum(c.assessment.true_positives for c in self.cases),
            "false_positives": sum(c.assessment.false_positives for c in self.cases),
            "false_negatives": sum(c.assessment.false_negatives for c in self.cases),
            "critical_failures": len(self.critical_failures),
        }


def training_examples(cases: tuple[CompiledFormationCase, ...]) -> tuple[CompiledFormationCase, ...]:
    """Expose only the person-separated train split to the optimizer."""
    train = tuple(case for case in cases if case.split == "train")
    if not train:
        raise CognitiveKernelContractError("no admitted formation training examples")
    return train


def evaluate_candidate(
    candidate: LearnedFormationCandidate,
    cases: tuple[CompiledFormationCase, ...], *, split: str,
) -> CandidateEvaluation:
    digest = require_sha256(candidate.artifact_sha256, "artifact_sha256")
    if not cases or split not in {"development", "challenge"}:
        raise CognitiveKernelContractError("candidate evaluation requires development or challenge gold")
    selected = tuple(case for case in cases if case.split == split)
    if not selected:
        raise CognitiveKernelContractError("requested gold split is empty")
    results = []
    for case in selected:
        opened = tuple((ref_id, text.encode("utf-8")) for ref_id, text in case.texts)
        output = candidate.infer(context=case.gold.context, opened_sources=opened)
        if output.model_artifact_digest != digest:
            raise CognitiveKernelContractError("candidate output cites different model artifact")
        assessment = assess_formation(case.gold, output)
        results.append(CandidateCaseResult(case.gold.case_id, split, assessment,
                                           output.content_digest()))
    return CandidateEvaluation(digest, tuple(results))
