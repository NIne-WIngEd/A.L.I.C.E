"""Controlled composition of P4.6a research into verified P4.6b evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime

from .contracts import InformationQuery, sha256_text
from .freshness import (
    DeterministicInformationFreshnessEvaluator,
    DeterministicInformationTemporalClassifier,
    InformationTemporalIntent,
    InformationTemporallyQualifiedSource,
)
from .freshness_policy import InformationFreshnessPolicy
from .grounding import (
    DeterministicInformationGroundingBuilder,
    InformationClaimDraft,
    InformationVerifiedGroundingPacket,
)
from .grounding_policy import InformationGroundingPolicy
from .injection_firewall import (
    DeterministicInformationInjectionFirewall,
    InformationInspectedSource,
)
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy
from .research_evidence_policy import InformationResearchEvidencePolicy
from .research_orchestration import InformationResearchRun
from .research_orchestration_policy import InformationResearchOrchestrationPolicy

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DISPOSITIONS = {"qualified", "blocked_injection", "freshness_rejected"}


class InformationResearchEvidenceError(RuntimeError):
    """Raised when a P4.6b pipeline result cannot be built or verified."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchEvidenceError(f"{field} must be non-empty text.")
    return value.strip()


def _digest(value: object, field: str) -> str:
    text = _text(value, field).lower()
    if _SHA256.fullmatch(text) is None:
        raise InformationResearchEvidenceError(f"{field} must be a lowercase SHA-256 digest.")
    return text


def _timestamp(value: object, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InformationResearchEvidenceError(f"{field} must be valid ISO-8601 text.") from exc
    if parsed.tzinfo is None:
        raise InformationResearchEvidenceError(f"{field} must include a timezone offset.")
    return text


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class InformationEvidenceSourceOutcome:
    """Metadata-safe result of applying the P4.3 and P4.4 gates to one source."""

    source_id: str
    canonical_url: str
    source_content_sha256: str
    inspection_verdict: str
    freshness_verdict: str | None
    supports_claim: bool
    disposition: str
    reason_code: str | None

    def validate(self) -> None:
        _text(self.source_id, "source_id")
        _text(self.canonical_url, "canonical_url")
        _digest(self.source_content_sha256, "source_content_sha256")
        if self.inspection_verdict not in {"clear", "blocked"}:
            raise InformationResearchEvidenceError("Inspection verdict is not recognized.")
        if self.freshness_verdict is not None:
            _text(self.freshness_verdict, "freshness_verdict")
        if not isinstance(self.supports_claim, bool):
            raise InformationResearchEvidenceError("supports_claim must be boolean.")
        if self.disposition not in _DISPOSITIONS:
            raise InformationResearchEvidenceError("Source disposition is not recognized.")
        if self.disposition == "qualified":
            if self.inspection_verdict != "clear" or self.freshness_verdict is None or not self.supports_claim or self.reason_code is not None:
                raise InformationResearchEvidenceError("Qualified source metadata is inconsistent.")
        elif self.disposition == "blocked_injection":
            if self.inspection_verdict != "blocked" or self.freshness_verdict is not None or self.supports_claim or self.reason_code != "prompt_injection_blocked":
                raise InformationResearchEvidenceError("Blocked-source metadata is inconsistent.")
        else:
            if self.inspection_verdict != "clear" or self.freshness_verdict is None or self.supports_claim or self.reason_code != "freshness_insufficient":
                raise InformationResearchEvidenceError("Freshness-rejected metadata is inconsistent.")

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "source_content_sha256": self.source_content_sha256,
            "inspection_verdict": self.inspection_verdict,
            "freshness_verdict": self.freshness_verdict,
            "supports_claim": self.supports_claim,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
        }


def _receipt_payload(receipt: "InformationResearchEvidenceReceipt", *, include_ids: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "pipeline_id": receipt.pipeline_id,
        "research_run_id": receipt.research_run_id,
        "research_receipt_sha256": receipt.research_receipt_sha256,
        "query_id": receipt.query_id,
        "query_content_sha256": receipt.query_content_sha256,
        "research_outcome": receipt.research_outcome,
        "research_stopping_reason": receipt.research_stopping_reason,
        "pipeline_outcome": receipt.pipeline_outcome,
        "partial_research": receipt.partial_research,
        "temporal_intent_id": receipt.temporal_intent_id,
        "temporal_intent_kind": receipt.temporal_intent_kind,
        "source_outcomes": [item.metadata_record() for item in receipt.source_outcomes],
        "qualified_source_ids": list(receipt.qualified_source_ids),
        "qualified_source_content_sha256s": list(receipt.qualified_source_content_sha256s),
        "grounding_sha256": receipt.grounding_sha256,
        "policy_versions": list(receipt.policy_versions),
        "created_at": receipt.created_at,
    }
    if not include_ids:
        payload.pop("pipeline_id")
    return payload


def _pipeline_id(receipt: "InformationResearchEvidenceReceipt") -> str:
    digest = hashlib.sha256(_canonical_json(_receipt_payload(receipt, include_ids=False)).encode("utf-8")).hexdigest()
    return f"evidence-{digest[:20]}"


def _receipt_sha256(receipt: "InformationResearchEvidenceReceipt") -> str:
    return hashlib.sha256(_canonical_json(_receipt_payload(receipt)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InformationResearchEvidenceReceipt:
    """Raw-content-free binding from one research run to one grounding packet."""

    pipeline_id: str
    research_run_id: str
    research_receipt_sha256: str
    query_id: str
    query_content_sha256: str
    research_outcome: str
    research_stopping_reason: str
    pipeline_outcome: str
    partial_research: bool
    temporal_intent_id: str
    temporal_intent_kind: str
    source_outcomes: tuple[InformationEvidenceSourceOutcome, ...]
    qualified_source_ids: tuple[str, ...]
    qualified_source_content_sha256s: tuple[str, ...]
    grounding_sha256: str
    policy_versions: tuple[str, ...]
    created_at: str
    receipt_sha256: str

    @classmethod
    def create(cls, **values: object) -> "InformationResearchEvidenceReceipt":
        draft = cls(pipeline_id="evidence-pending", receipt_sha256="0" * 64, **values)  # type: ignore[arg-type]
        identified = cls(**{**draft.__dict__, "pipeline_id": _pipeline_id(draft)})
        receipt = cls(**{**identified.__dict__, "receipt_sha256": _receipt_sha256(identified)})
        receipt.validate()
        return receipt

    def validate(self) -> None:
        _text(self.pipeline_id, "pipeline_id")
        _text(self.research_run_id, "research_run_id")
        _digest(self.research_receipt_sha256, "research_receipt_sha256")
        _text(self.query_id, "query_id")
        _digest(self.query_content_sha256, "query_content_sha256")
        if self.research_outcome not in {"completed", "partial"}:
            raise InformationResearchEvidenceError("Research outcome is not eligible for P4.6b.")
        _text(self.research_stopping_reason, "research_stopping_reason")
        if self.pipeline_outcome not in {"answerable", "conflict", "uncertain", "insufficient_sources"}:
            raise InformationResearchEvidenceError("Pipeline outcome is not recognized.")
        if self.partial_research is not (self.research_outcome == "partial"):
            raise InformationResearchEvidenceError("Partial-research marker is inconsistent.")
        if self.partial_research and self.pipeline_outcome == "answerable":
            raise InformationResearchEvidenceError("Partial research cannot be promoted to answerable.")
        _text(self.temporal_intent_id, "temporal_intent_id")
        _text(self.temporal_intent_kind, "temporal_intent_kind")
        seen_source_ids: set[str] = set()
        for outcome in self.source_outcomes:
            outcome.validate()
            if outcome.source_id in seen_source_ids:
                raise InformationResearchEvidenceError("Source outcomes cannot contain duplicates.")
            seen_source_ids.add(outcome.source_id)
        if len(self.qualified_source_ids) != len(self.qualified_source_content_sha256s):
            raise InformationResearchEvidenceError("Qualified source IDs and digests must align.")
        if len(set(self.qualified_source_ids)) != len(self.qualified_source_ids):
            raise InformationResearchEvidenceError("Qualified source IDs cannot contain duplicates.")
        for digest in self.qualified_source_content_sha256s:
            _digest(digest, "qualified_source_content_sha256")
        expected_qualified = tuple(item.source_id for item in self.source_outcomes if item.disposition == "qualified")
        expected_digests = tuple(item.source_content_sha256 for item in self.source_outcomes if item.disposition == "qualified")
        if self.qualified_source_ids != expected_qualified or self.qualified_source_content_sha256s != expected_digests:
            raise InformationResearchEvidenceError("Qualified source sequence does not match source outcomes.")
        _digest(self.grounding_sha256, "grounding_sha256")
        if len(self.policy_versions) != 6 or any(not isinstance(value, str) or "@" not in value for value in self.policy_versions):
            raise InformationResearchEvidenceError("Policy-version bindings are incomplete.")
        _timestamp(self.created_at, "created_at")
        if self.pipeline_id != _pipeline_id(self):
            raise InformationResearchEvidenceError("Pipeline ID does not match complete receipt metadata.")
        if _digest(self.receipt_sha256, "receipt_sha256") != _receipt_sha256(self):
            raise InformationResearchEvidenceError("Evidence receipt digest does not match metadata.")

    def to_metadata_record(self) -> dict[str, object]:
        self.validate()
        return {**_receipt_payload(self), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class InformationResearchEvidenceResult:
    """Verified in-memory gate outputs plus a metadata-safe P4.6b receipt."""

    research_run: InformationResearchRun
    temporal_intent: InformationTemporalIntent
    inspected_sources: tuple[InformationInspectedSource, ...]
    qualified_sources: tuple[InformationTemporallyQualifiedSource, ...]
    grounding: InformationVerifiedGroundingPacket
    receipt: InformationResearchEvidenceReceipt

    def validate(self, *, pipeline: "DeterministicInformationResearchEvidencePipeline") -> None:
        pipeline._validate_result(self)


@dataclass(frozen=True)
class DeterministicInformationResearchEvidencePipeline:
    """Compose exact P4.6a, P4.3, P4.4a and P4.5a boundaries."""

    information_policy: InformationPolicy
    orchestration_policy: InformationResearchOrchestrationPolicy
    firewall_policy: InformationInjectionFirewallPolicy
    freshness_policy: InformationFreshnessPolicy
    grounding_policy: InformationGroundingPolicy
    evidence_policy: InformationResearchEvidencePolicy

    def __post_init__(self) -> None:
        self.evidence_policy.validate(
            information_policy=self.information_policy,
            orchestration_policy=self.orchestration_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
            grounding_policy=self.grounding_policy,
        )

    def process(
        self,
        *,
        research_run: InformationResearchRun,
        reference_time: str,
        outcome: str,
        claim_drafts: tuple[InformationClaimDraft, ...],
        created_at: str,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> InformationResearchEvidenceResult:
        research_run.validate(policy=self.orchestration_policy)
        if research_run.receipt.outcome not in self.evidence_policy.allowed_research_input_outcomes:
            raise InformationResearchEvidenceError("Research run has no preserved sources eligible for P4.6b.")
        if len(research_run.sources) > self.evidence_policy.max_sources or len(claim_drafts) > self.evidence_policy.max_claims:
            raise InformationResearchEvidenceError("P4.6b evidence budget exceeded.")
        if research_run.receipt.outcome == "partial" and outcome == "answerable":
            raise InformationResearchEvidenceError("Partial research cannot be promoted to answerable.")
        query = research_run.request.query
        classifier = DeterministicInformationTemporalClassifier(self.freshness_policy)
        intent = classifier.classify(
            query,
            reference_time=reference_time,
            window_start=window_start,
            window_end=window_end,
        )
        firewall = DeterministicInformationInjectionFirewall(
            self.information_policy,
            self.firewall_policy,
        )
        freshness = DeterministicInformationFreshnessEvaluator(
            self.information_policy,
            self.firewall_policy,
            self.freshness_policy,
        )
        inspected: list[InformationInspectedSource] = []
        qualified: list[InformationTemporallyQualifiedSource] = []
        source_outcomes: list[InformationEvidenceSourceOutcome] = []
        for source in research_run.sources:
            inspected_source = firewall.inspect(source)
            inspected.append(inspected_source)
            if inspected_source.inspection.verdict == "blocked":
                source_outcomes.append(
                    InformationEvidenceSourceOutcome(
                        source_id=source.source_id,
                        canonical_url=source.canonical_url,
                        source_content_sha256=source.content_sha256,
                        inspection_verdict="blocked",
                        freshness_verdict=None,
                        supports_claim=False,
                        disposition="blocked_injection",
                        reason_code="prompt_injection_blocked",
                    )
                )
                continue
            assessed = freshness.assess(inspected_source, intent=intent, query=query)
            if assessed.assessment.supports_claim:
                qualified.append(assessed)
                disposition = "qualified"
                reason = None
            else:
                disposition = "freshness_rejected"
                reason = "freshness_insufficient"
            source_outcomes.append(
                InformationEvidenceSourceOutcome(
                    source_id=source.source_id,
                    canonical_url=source.canonical_url,
                    source_content_sha256=source.content_sha256,
                    inspection_verdict="clear",
                    freshness_verdict=assessed.assessment.verdict,
                    supports_claim=assessed.assessment.supports_claim,
                    disposition=disposition,
                    reason_code=reason,
                )
            )
        if qualified:
            if outcome == "insufficient_sources":
                raise InformationResearchEvidenceError("Qualified sources cannot be discarded as insufficient.")
        else:
            if outcome != "insufficient_sources" or claim_drafts:
                raise InformationResearchEvidenceError("No qualified sources require an empty insufficient-sources grounding.")
        builder = DeterministicInformationGroundingBuilder(
            self.information_policy,
            self.firewall_policy,
            self.freshness_policy,
            self.grounding_policy,
        )
        grounding = builder.build(
            packet_id=f"grounding-{research_run.receipt.run_id}",
            request_id=research_run.request.request_id,
            outcome=outcome,
            query=query,
            qualified_sources=tuple(qualified),
            claim_drafts=claim_drafts,
            created_at=created_at,
        )
        receipt = InformationResearchEvidenceReceipt.create(
            research_run_id=research_run.receipt.run_id,
            research_receipt_sha256=research_run.receipt.receipt_sha256,
            query_id=query.query_id,
            query_content_sha256=query.content_sha256,
            research_outcome=research_run.receipt.outcome,
            research_stopping_reason=research_run.receipt.stopping_reason,
            pipeline_outcome=grounding.packet.outcome,
            partial_research=research_run.receipt.outcome == "partial",
            temporal_intent_id=intent.intent_id,
            temporal_intent_kind=intent.kind,
            source_outcomes=tuple(source_outcomes),
            qualified_source_ids=tuple(item.inspected_source.source.source_id for item in qualified),
            qualified_source_content_sha256s=tuple(item.inspected_source.source.content_sha256 for item in qualified),
            grounding_sha256=grounding.grounding_sha256,
            policy_versions=self._policy_versions(),
            created_at=created_at,
        )
        result = InformationResearchEvidenceResult(
            research_run=research_run,
            temporal_intent=intent,
            inspected_sources=tuple(inspected),
            qualified_sources=tuple(qualified),
            grounding=grounding,
            receipt=receipt,
        )
        self._validate_result(result)
        return result

    def _policy_versions(self) -> tuple[str, ...]:
        policies = (
            self.information_policy,
            self.orchestration_policy,
            self.firewall_policy,
            self.freshness_policy,
            self.grounding_policy,
            self.evidence_policy,
        )
        return tuple(f"{item.policy_name}@{item.version}" for item in policies)

    def _validate_result(self, result: InformationResearchEvidenceResult) -> None:
        self.evidence_policy.validate(
            information_policy=self.information_policy,
            orchestration_policy=self.orchestration_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
            grounding_policy=self.grounding_policy,
        )
        result.research_run.validate(policy=self.orchestration_policy)
        query: InformationQuery = result.research_run.request.query
        result.temporal_intent.validate(query=query, policy=self.freshness_policy)
        if len(result.inspected_sources) != len(result.research_run.sources):
            raise InformationResearchEvidenceError("Every preserved source requires one inspection.")
        firewall = DeterministicInformationInjectionFirewall(self.information_policy, self.firewall_policy)
        freshness = DeterministicInformationFreshnessEvaluator(self.information_policy, self.firewall_policy, self.freshness_policy)
        derived_outcomes: list[InformationEvidenceSourceOutcome] = []
        derived_qualified: list[InformationTemporallyQualifiedSource] = []
        for source, supplied in zip(result.research_run.sources, result.inspected_sources):
            expected = firewall.inspect(source)
            if supplied != expected:
                raise InformationResearchEvidenceError("Source inspection does not match the exact source version.")
            if expected.inspection.verdict == "blocked":
                derived_outcomes.append(InformationEvidenceSourceOutcome(
                    source_id=source.source_id,
                    canonical_url=source.canonical_url,
                    source_content_sha256=source.content_sha256,
                    inspection_verdict="blocked",
                    freshness_verdict=None,
                    supports_claim=False,
                    disposition="blocked_injection",
                    reason_code="prompt_injection_blocked",
                ))
                continue
            assessed = freshness.assess(expected, intent=result.temporal_intent, query=query)
            if assessed.assessment.supports_claim:
                derived_qualified.append(assessed)
                disposition, reason = "qualified", None
            else:
                disposition, reason = "freshness_rejected", "freshness_insufficient"
            derived_outcomes.append(InformationEvidenceSourceOutcome(
                source_id=source.source_id,
                canonical_url=source.canonical_url,
                source_content_sha256=source.content_sha256,
                inspection_verdict="clear",
                freshness_verdict=assessed.assessment.verdict,
                supports_claim=assessed.assessment.supports_claim,
                disposition=disposition,
                reason_code=reason,
            ))
        if tuple(derived_qualified) != result.qualified_sources:
            raise InformationResearchEvidenceError("Qualified-source set does not match gate results.")
        result.grounding.validate(
            query=query,
            qualified_sources=result.qualified_sources,
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
            grounding_policy=self.grounding_policy,
        )
        result.receipt.validate()
        if result.receipt.research_run_id != result.research_run.receipt.run_id or result.receipt.research_receipt_sha256 != result.research_run.receipt.receipt_sha256:
            raise InformationResearchEvidenceError("Research receipt binding does not match.")
        if result.receipt.query_id != query.query_id or result.receipt.query_content_sha256 != query.content_sha256:
            raise InformationResearchEvidenceError("Query binding does not match.")
        if result.receipt.research_outcome != result.research_run.receipt.outcome or result.receipt.research_stopping_reason != result.research_run.receipt.stopping_reason:
            raise InformationResearchEvidenceError("Research completion metadata does not match.")
        if result.receipt.pipeline_outcome != result.grounding.packet.outcome or result.receipt.grounding_sha256 != result.grounding.grounding_sha256:
            raise InformationResearchEvidenceError("Grounding binding does not match.")
        if result.receipt.temporal_intent_id != result.temporal_intent.intent_id or result.receipt.temporal_intent_kind != result.temporal_intent.kind:
            raise InformationResearchEvidenceError("Temporal-intent binding does not match.")
        if result.receipt.source_outcomes != tuple(derived_outcomes):
            raise InformationResearchEvidenceError("Source outcome metadata does not match gate results.")
        if result.receipt.policy_versions != self._policy_versions():
            raise InformationResearchEvidenceError("Policy-version bindings do not match selected policies.")
