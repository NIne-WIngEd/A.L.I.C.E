from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from alice_information.research_execution import (
    DeterministicInformationResearchExecution,
    InformationResearchExecutionError,
    InformationResearchExecutionPlan,
)
from alice_information.research_execution_policy import (
    parse_information_research_execution_policy,
)

NOW = "2026-07-29T11:44:00-05:00"
DIGEST = hashlib.sha256(b"fixture").hexdigest()


def policy_payload() -> dict[str, object]:
    return {
        "policy_name": "alice_information_research_execution_policy",
        "version": "1.0.0",
        "phase": "4",
        "milestone": "P4.7b",
        "status": "governed_research_execution",
        "permission_id": "web.search",
        "allowed_modes": ["local_only", "research"],
        "allowed_requested_availability_states": ["not_requested", "available", "offline", "unavailable"],
        "allowed_result_availability_states": ["not_requested", "available", "offline", "unavailable"],
        "allowed_result_statuses": ["completed", "unavailable"],
        "allowed_unavailable_reasons": ["offline", "unavailable", "research_cancelled", "research_failed", "insufficient_sources", "insufficient_evidence"],
        "explicit_mode_required": True,
        "exact_provider_selection_required": True,
        "deterministic_fixture_execution_required": True,
        "orchestration_revalidation_required": True,
        "evidence_revalidation_required": True,
        "mode_adapter_revalidation_required": True,
        "preconversation_failure_handling_required": True,
        "local_only_provider_execution_allowed": False,
        "silent_web_activation_allowed": False,
        "provider_fallback_allowed": False,
        "live_provider_registration_allowed": False,
        "source_body_persistence_allowed": False,
        "memory_write_allowed": False,
        "phase5_storage_runtime_allowed": False,
        "external_action_allowed": False,
        "retry_allowed": False,
        "recursive_browsing_allowed": False,
        "background_execution_allowed": False,
    }


class FakeOrchestrationPolicy:
    policy_name = "alice_information_research_orchestration_policy"
    version = "1.0.0"
    deterministic_fixture_only = True
    provider_fallback_allowed = False
    live_provider_registration_allowed = False

    def validate(self) -> None:
        return None


class FakeEvidencePolicy:
    policy_name = "alice_information_research_evidence_policy"
    version = "1.0.0"

    def validate(self, **kwargs) -> None:
        return None


class FakeModePolicy:
    policy_name = "alice_information_research_mode_policy"
    version = "1.0.0"

    def validate(self, **kwargs) -> None:
        return None


class FakeQuery:
    query_id = "query-1"
    content_sha256 = hashlib.sha256(b"public minimized query").hexdigest()
    text = "public minimized query"


class FakeRequest:
    request_id = "research-request-1"
    query = FakeQuery()

    def validate(self) -> None:
        return None


class FakeDraft:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid

    def validate_shape(self) -> None:
        if not self.valid:
            raise ValueError("invalid draft")


class FakeCommand:
    session_id = "session-1"
    turn_id = "turn-1"
    request_id = "conversation-request-1"
    generation_id = "generation-1"

    def __init__(self, *, grounding=None) -> None:
        self.grounding = grounding

    def validate(self) -> None:
        return None


class FakeRunReceipt:
    def __init__(self, outcome: str, *, search="search-fixture", fetch="fetch-fixture") -> None:
        self.run_id = f"run-{outcome}"
        self.receipt_sha256 = hashlib.sha256(self.run_id.encode()).hexdigest()
        self.outcome = outcome
        self.stopping_reason = {
            "completed": "all_selected_sources_fetched",
            "partial": "partial_fetch_failure",
            "insufficient_sources": "all_fetches_failed",
            "cancelled": "cancelled",
            "failed": "search_failed",
        }[outcome]
        self.search_provider = search
        self.fetch_provider = fetch


class FakeRun:
    def __init__(self, request, outcome="completed", *, search="search-fixture", fetch="fetch-fixture") -> None:
        self.request = request
        self.receipt = FakeRunReceipt(outcome, search=search, fetch=fetch)
        self.validate_calls = 0

    def validate(self, *, policy) -> None:
        self.validate_calls += 1


class FakePacket:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome


class FakeGrounding:
    def __init__(self, outcome: str) -> None:
        self.packet = FakePacket(outcome)


class FakeEvidenceReceipt:
    def __init__(self, outcome: str) -> None:
        self.pipeline_id = f"evidence-{outcome}"
        self.receipt_sha256 = hashlib.sha256(self.pipeline_id.encode()).hexdigest()


class FakeEvidence:
    def __init__(self, research_run: FakeRun, outcome: str) -> None:
        self.research_run = research_run
        self.grounding = FakeGrounding(outcome)
        self.receipt = FakeEvidenceReceipt(outcome)
        self.validate_calls = 0

    def validate(self, *, pipeline) -> None:
        self.validate_calls += 1

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FakeEvidence)
            and self.research_run is other.research_run
            and self.grounding.packet.outcome == other.grounding.packet.outcome
            and self.receipt.pipeline_id == other.receipt.pipeline_id
            and self.receipt.receipt_sha256 == other.receipt.receipt_sha256
        )


class FakeModeReceipt:
    def __init__(self, mode: str, availability: str, status: str) -> None:
        self.adapter_id = f"adapter-{mode}-{availability}-{status}"
        self.receipt_sha256 = hashlib.sha256(self.adapter_id.encode()).hexdigest()


class FakeModeResult:
    def __init__(self, mode: str, availability: str, status: str) -> None:
        self.mode = mode
        self.availability = availability
        self.status = status
        self.receipt = FakeModeReceipt(mode, availability, status)
        self.validate_calls = 0

    def validate(self, *, adapter, command, evidence_result) -> None:
        self.validate_calls += 1
        if self.status == "unavailable" and evidence_result is not None:
            raise AssertionError("unavailable mode result received evidence")
        if self.status == "completed" and self.mode == "research" and evidence_result is None:
            raise AssertionError("completed research result missed evidence")


class RecordingOrchestrator:
    def __init__(self, policy, *, outcome="completed", substitute_request=False, substitute_provider=False) -> None:
        self.policy = policy
        self.outcome = outcome
        self.substitute_request = substitute_request
        self.substitute_provider = substitute_provider
        self.calls = []

    def execute(self, request, *, search_provider, fetch_provider, cancellation=None):
        self.calls.append((request, search_provider, fetch_provider, cancellation))
        selected_request = FakeRequest() if self.substitute_request else request
        if self.substitute_request:
            selected_request.request_id = "substituted"  # type: ignore[misc]
        selected_search = "other-search" if self.substitute_provider else search_provider
        return FakeRun(selected_request, self.outcome, search=selected_search, fetch=fetch_provider)


class RecordingEvidencePipeline:
    def __init__(self, orchestration_policy, evidence_policy, *, result_outcome="answerable") -> None:
        self.orchestration_policy = orchestration_policy
        self.evidence_policy = evidence_policy
        self.result_outcome = result_outcome
        self.calls = []

    def process(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["research_run"].receipt.outcome == "partial" and kwargs["outcome"] == "answerable":
            raise ValueError("partial research cannot be answerable")
        return FakeEvidence(kwargs["research_run"], self.result_outcome)


class RecordingModeAdapter:
    def __init__(self, evidence_pipeline, mode_policy) -> None:
        self.evidence_pipeline = evidence_pipeline
        self.mode_policy = mode_policy
        self.calls = []

    def run_turn(self, command, *, mode, availability, evidence_result=None, cancellation=None):
        self.calls.append((command, mode, availability, evidence_result, cancellation))
        status = "unavailable" if availability in {"offline", "unavailable"} else "completed"
        return FakeModeResult(mode, availability, status)


def make_executor(*, run_outcome="completed", evidence_outcome="answerable", **orchestrator_kwargs):
    orchestration_policy = FakeOrchestrationPolicy()
    evidence_policy = FakeEvidencePolicy()
    mode_policy = FakeModePolicy()
    orchestration = RecordingOrchestrator(
        orchestration_policy,
        outcome=run_outcome,
        **orchestrator_kwargs,
    )
    evidence = RecordingEvidencePipeline(
        orchestration_policy,
        evidence_policy,
        result_outcome=evidence_outcome,
    )
    adapter = RecordingModeAdapter(evidence, mode_policy)
    policy = parse_information_research_execution_policy(
        policy_payload(),
        orchestration_policy=orchestration_policy,
        evidence_policy=evidence_policy,
        mode_policy=mode_policy,
    )
    executor = DeterministicInformationResearchExecution(
        research_orchestrator=orchestration,
        evidence_pipeline=evidence,
        mode_adapter=adapter,
        execution_policy=policy,
        clock=lambda: NOW,
    )
    return executor, orchestration, evidence, adapter


def local_plan() -> InformationResearchExecutionPlan:
    return InformationResearchExecutionPlan(mode="local_only", availability="not_requested")


def research_plan(*, outcome="answerable", availability="available", drafts=None):
    if availability != "available":
        return InformationResearchExecutionPlan(mode="research", availability=availability)
    if drafts is None:
        drafts = () if outcome == "insufficient_sources" else (FakeDraft(),)
    return InformationResearchExecutionPlan(
        mode="research",
        availability="available",
        research_request=FakeRequest(),
        search_provider="search-fixture",
        fetch_provider="fetch-fixture",
        reference_time=NOW,
        evidence_outcome=outcome,
        claim_drafts=drafts,
        created_at=NOW,
    )


def test_local_only_never_executes_research() -> None:
    executor, orchestration, evidence, adapter = make_executor()
    result = executor.run_turn(FakeCommand(), plan=local_plan())
    assert result.status == "completed"
    assert result.availability == "not_requested"
    assert orchestration.calls == []
    assert evidence.calls == []
    assert adapter.calls[0][1:4] == ("local_only", "not_requested", None)


@pytest.mark.parametrize("availability", ["offline", "unavailable"])
def test_preflight_unavailable_never_executes_providers(availability: str) -> None:
    executor, orchestration, evidence, adapter = make_executor()
    result = executor.run_turn(FakeCommand(), plan=research_plan(availability=availability))
    assert result.status == "unavailable"
    assert result.unavailable_reason == availability
    assert orchestration.calls == []
    assert evidence.calls == []
    assert adapter.calls[0][2] == availability


def test_completed_execution_composes_all_boundaries() -> None:
    executor, orchestration, evidence, adapter = make_executor()
    plan = research_plan()
    result = executor.run_turn(FakeCommand(), plan=plan, cancellation="token")
    assert result.status == "completed"
    assert result.availability == "available"
    assert orchestration.calls[0][1:3] == ("search-fixture", "fetch-fixture")
    assert orchestration.calls[0][3] == "token"
    assert evidence.calls[0]["research_run"] is result.research_run
    assert evidence.calls[0]["claim_drafts"] == plan.claim_drafts
    assert adapter.calls[0][1:4] == ("research", "available", result.evidence_result)
    result.validate(executor=executor, command=FakeCommand(), plan=plan)


def test_partial_uncertain_execution_is_preserved() -> None:
    executor, _, _, _ = make_executor(run_outcome="partial", evidence_outcome="uncertain")
    plan = research_plan(outcome="uncertain")
    result = executor.run_turn(FakeCommand(), plan=plan)
    assert result.status == "completed"
    assert result.receipt.research_outcome == "partial"
    assert result.receipt.evidence_outcome == "uncertain"


def test_partial_answerable_is_rejected_before_conversation() -> None:
    executor, _, _, adapter = make_executor(run_outcome="partial")
    with pytest.raises(ValueError, match="partial research"):
        executor.run_turn(FakeCommand(), plan=research_plan())
    assert adapter.calls == []


@pytest.mark.parametrize(
    "run_outcome,reason",
    [
        ("insufficient_sources", "insufficient_sources"),
        ("cancelled", "research_cancelled"),
        ("failed", "research_failed"),
    ],
)
def test_run_level_failure_returns_before_conversation_mutation(run_outcome: str, reason: str) -> None:
    executor, orchestration, evidence, adapter = make_executor(run_outcome=run_outcome)
    result = executor.run_turn(FakeCommand(), plan=research_plan())
    assert orchestration.calls
    assert evidence.calls == []
    assert result.status == "unavailable"
    assert result.availability == "unavailable"
    assert result.unavailable_reason == reason
    assert result.research_run is not None
    assert result.evidence_result is None
    assert adapter.calls[0][1:4] == ("research", "unavailable", None)


def test_insufficient_qualified_evidence_returns_unavailable() -> None:
    executor, _, evidence, adapter = make_executor(evidence_outcome="insufficient_sources")
    plan = research_plan(outcome="insufficient_sources")
    result = executor.run_turn(FakeCommand(), plan=plan)
    assert evidence.calls
    assert result.status == "unavailable"
    assert result.unavailable_reason == "insufficient_evidence"
    assert result.evidence_result is not None
    assert adapter.calls[0][3] is None
    assert result.receipt.evidence_outcome == "insufficient_sources"


def test_research_rejects_preinjected_grounding_before_provider_use() -> None:
    executor, orchestration, _, _ = make_executor()
    with pytest.raises(InformationResearchExecutionError) as error:
        executor.run_turn(FakeCommand(grounding=object()), plan=research_plan())
    assert error.value.code == "research_execution_grounding_preinjected"
    assert orchestration.calls == []


def test_provider_substitution_is_rejected_before_evidence() -> None:
    executor, _, evidence, adapter = make_executor(substitute_provider=True)
    with pytest.raises(InformationResearchExecutionError) as error:
        executor.run_turn(FakeCommand(), plan=research_plan())
    assert error.value.code == "research_execution_provider_substituted"
    assert evidence.calls == []
    assert adapter.calls == []


def test_request_substitution_is_rejected_before_evidence() -> None:
    executor, _, evidence, adapter = make_executor(substitute_request=True)
    with pytest.raises(InformationResearchExecutionError, match="substituted"):
        executor.run_turn(FakeCommand(), plan=research_plan())
    assert evidence.calls == []
    assert adapter.calls == []


@pytest.mark.parametrize(
    "plan",
    [
        InformationResearchExecutionPlan(mode="local_only", availability="available"),
        InformationResearchExecutionPlan(mode="local_only", availability="not_requested", research_request=FakeRequest()),
        InformationResearchExecutionPlan(mode="research", availability="not_requested"),
        InformationResearchExecutionPlan(mode="unknown", availability="not_requested"),
    ],
)
def test_invalid_mode_state_is_rejected_before_execution(plan) -> None:
    executor, orchestration, _, _ = make_executor()
    with pytest.raises(InformationResearchExecutionError):
        executor.run_turn(FakeCommand(), plan=plan)
    assert orchestration.calls == []


@pytest.mark.parametrize("provider", ["UPPER", "bad provider", "", "x/evil"])
def test_invalid_provider_identity_is_rejected(provider: str) -> None:
    executor, orchestration, _, _ = make_executor()
    plan = replace(research_plan(), search_provider=provider)
    with pytest.raises(InformationResearchExecutionError) as error:
        executor.run_turn(FakeCommand(), plan=plan)
    assert error.value.code == "research_execution_provider_invalid"
    assert orchestration.calls == []


@pytest.mark.parametrize("field", ["reference_time", "created_at", "window_start", "window_end"])
def test_invalid_temporal_plan_is_rejected(field: str) -> None:
    executor, orchestration, _, _ = make_executor()
    plan = replace(research_plan(), **{field: "not-a-time"})
    with pytest.raises(InformationResearchExecutionError):
        executor.run_turn(FakeCommand(), plan=plan)
    assert orchestration.calls == []


def test_claim_outcome_requires_claims() -> None:
    executor, orchestration, _, _ = make_executor()
    with pytest.raises(InformationResearchExecutionError):
        executor.run_turn(FakeCommand(), plan=research_plan(drafts=()))
    assert orchestration.calls == []


def test_insufficient_sources_forbids_claims() -> None:
    executor, orchestration, _, _ = make_executor()
    plan = research_plan(outcome="insufficient_sources", drafts=(FakeDraft(),))
    with pytest.raises(InformationResearchExecutionError):
        executor.run_turn(FakeCommand(), plan=plan)
    assert orchestration.calls == []


def test_invalid_claim_draft_is_rejected() -> None:
    executor, orchestration, _, _ = make_executor()
    with pytest.raises(ValueError, match="invalid draft"):
        executor.run_turn(FakeCommand(), plan=research_plan(drafts=(FakeDraft(valid=False),)))
    assert orchestration.calls == []


def test_claim_drafts_must_be_tuple() -> None:
    executor, orchestration, _, _ = make_executor()
    plan = replace(research_plan(), claim_drafts=[FakeDraft()])  # type: ignore[arg-type]
    with pytest.raises(InformationResearchExecutionError):
        executor.run_turn(FakeCommand(), plan=plan)
    assert orchestration.calls == []


def test_unavailable_preflight_forbids_execution_inputs() -> None:
    executor, orchestration, _, _ = make_executor()
    plan = replace(research_plan(availability="offline"), research_request=FakeRequest())
    with pytest.raises(InformationResearchExecutionError):
        executor.run_turn(FakeCommand(), plan=plan)
    assert orchestration.calls == []


def test_constructor_requires_shared_orchestration_policy() -> None:
    executor, orchestration, evidence, adapter = make_executor()
    evidence.orchestration_policy = FakeOrchestrationPolicy()
    with pytest.raises(InformationResearchExecutionError, match="exact policy"):
        DeterministicInformationResearchExecution(
            research_orchestrator=orchestration,
            evidence_pipeline=evidence,
            mode_adapter=adapter,
            execution_policy=executor.execution_policy,
            clock=lambda: NOW,
        )


def test_constructor_requires_shared_evidence_pipeline() -> None:
    executor, orchestration, evidence, adapter = make_executor()
    adapter.evidence_pipeline = object()
    with pytest.raises(InformationResearchExecutionError, match="exact selected evidence"):
        DeterministicInformationResearchExecution(
            research_orchestrator=orchestration,
            evidence_pipeline=evidence,
            mode_adapter=adapter,
            execution_policy=executor.execution_policy,
            clock=lambda: NOW,
        )


@pytest.mark.parametrize("missing", ["execute", "process", "run_turn"])
def test_constructor_requires_each_boundary(missing: str) -> None:
    executor, orchestration, evidence, adapter = make_executor()
    selected = {
        "execute": (object(), evidence, adapter),
        "process": (orchestration, object(), adapter),
        "run_turn": (orchestration, evidence, object()),
    }[missing]
    with pytest.raises(InformationResearchExecutionError):
        DeterministicInformationResearchExecution(
            research_orchestrator=selected[0],
            evidence_pipeline=selected[1],
            mode_adapter=selected[2],
            execution_policy=executor.execution_policy,
            clock=lambda: NOW,
        )


def test_receipt_contains_no_raw_query_or_claim_text() -> None:
    executor, _, _, _ = make_executor()
    result = executor.run_turn(FakeCommand(), plan=research_plan())
    rendered = str(result.receipt.to_metadata_record())
    assert FakeQuery.text not in rendered
    assert "source body" not in rendered
    assert result.receipt.query_sha256 == FakeQuery.content_sha256


def test_receipt_binds_selected_policy_versions() -> None:
    executor, _, _, _ = make_executor()
    result = executor.run_turn(FakeCommand(), plan=research_plan())
    assert result.receipt.policy_versions == (
        "alice_information_research_execution_policy@1.0.0",
        "alice_information_research_orchestration_policy@1.0.0",
        "alice_information_research_evidence_policy@1.0.0",
        "alice_information_research_mode_policy@1.0.0",
    )


def test_receipt_digest_tampering_is_rejected() -> None:
    executor, _, _, _ = make_executor()
    result = executor.run_turn(FakeCommand(), plan=research_plan())
    tampered = replace(result.receipt, receipt_sha256="0" * 64)
    with pytest.raises(InformationResearchExecutionError, match="digest"):
        tampered.validate()


def test_receipt_provider_tampering_is_rejected() -> None:
    executor, _, _, _ = make_executor()
    result = executor.run_turn(FakeCommand(), plan=research_plan())
    tampered_receipt = replace(result.receipt, search_provider="other")
    tampered_result = replace(result, receipt=tampered_receipt)
    with pytest.raises(InformationResearchExecutionError):
        tampered_result.validate(executor=executor, command=FakeCommand(), plan=research_plan())


def test_evidence_substitution_is_rejected() -> None:
    executor, _, _, _ = make_executor()
    plan = research_plan()
    result = executor.run_turn(FakeCommand(), plan=plan)
    substituted = FakeEvidence(result.research_run, "uncertain")
    tampered = replace(result, evidence_result=substituted)
    with pytest.raises(InformationResearchExecutionError):
        tampered.validate(executor=executor, command=FakeCommand(), plan=plan)


def test_run_substitution_is_rejected() -> None:
    executor, _, _, _ = make_executor()
    plan = research_plan()
    result = executor.run_turn(FakeCommand(), plan=plan)
    tampered = replace(result, research_run=FakeRun(plan.research_request, "completed"))
    with pytest.raises(InformationResearchExecutionError):
        tampered.validate(executor=executor, command=FakeCommand(), plan=plan)


def test_mode_receipt_substitution_is_rejected() -> None:
    executor, _, _, _ = make_executor()
    plan = research_plan()
    result = executor.run_turn(FakeCommand(), plan=plan)
    result.turn_result.receipt = FakeModeReceipt("research", "available", "completed")
    result.turn_result.receipt.adapter_id = "substituted"
    with pytest.raises(InformationResearchExecutionError):
        result.validate(executor=executor, command=FakeCommand(), plan=plan)


def test_validation_reprocesses_exact_evidence_plan() -> None:
    executor, _, evidence, _ = make_executor()
    plan = research_plan()
    result = executor.run_turn(FakeCommand(), plan=plan)
    initial = len(evidence.calls)
    result.validate(executor=executor, command=FakeCommand(), plan=plan)
    assert len(evidence.calls) == initial + 1


def test_plan_substitution_is_rejected() -> None:
    executor, _, _, _ = make_executor()
    plan = research_plan()
    result = executor.run_turn(FakeCommand(), plan=plan)
    different = replace(plan, fetch_provider="other-fetch")
    with pytest.raises(InformationResearchExecutionError):
        result.validate(executor=executor, command=FakeCommand(), plan=different)


def test_mode_adapter_receives_cancellation_on_fail_clean_path() -> None:
    executor, _, _, adapter = make_executor(run_outcome="cancelled")
    executor.run_turn(FakeCommand(), plan=research_plan(), cancellation="cancel-token")
    assert adapter.calls[0][4] == "cancel-token"
