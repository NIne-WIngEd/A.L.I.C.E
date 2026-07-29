from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from alice_conversation.contracts import (
    ConversationCitation,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    ConversationMessage,
    ModelResponse,
    sha256_text,
)
from alice_conversation.grounding_bridge import conversation_grounding_packet_sha256
from alice_conversation.orchestration import ConversationTurnCommand, ConversationTurnResult
from alice_conversation.response_validation_policy import (
    load_conversation_response_validation_policy,
)
from alice_information.contracts import (
    InformationActivityRecord,
    InformationQuery,
    InformationResearchRequest,
    InformationSearchResult,
    InformationSourceDocument,
)
from alice_information.conversation_bridge_policy import (
    load_information_conversation_bridge_policy,
)
from alice_information.freshness_policy import load_information_freshness_policy
from alice_information.grounding import InformationClaimDraft, InformationSupportSpan
from alice_information.grounding_policy import load_information_grounding_policy
from alice_information.injection_policy import load_information_injection_firewall_policy
from alice_information.policy import load_information_policy
from alice_information.research_evidence import (
    DeterministicInformationResearchEvidencePipeline,
)
from alice_information.research_evidence_policy import (
    load_information_research_evidence_policy,
)
from alice_information.research_mode import (
    DeterministicInformationResearchModeAdapter,
    InformationResearchModeError,
)
from alice_information.research_mode_policy import load_information_research_mode_policy
from alice_information.research_orchestration import (
    InformationResearchRun,
    InformationResearchRunReceipt,
)
from alice_information.research_orchestration_policy import (
    load_information_research_orchestration_policy,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-07-29T04:00:00Z"
CLAIM = "The launch date is July 30, 2026."


def information_policies():
    info = load_information_policy(ROOT / "policies/information_policy.json")
    orchestration = load_information_research_orchestration_policy(
        ROOT / "policies/information_research_orchestration_policy.json"
    )
    firewall = load_information_injection_firewall_policy(
        ROOT / "policies/information_injection_firewall_policy.json",
        information_policy=info,
    )
    freshness = load_information_freshness_policy(
        ROOT / "policies/information_freshness_policy.json",
        information_policy=info,
        firewall_policy=firewall,
    )
    grounding = load_information_grounding_policy(
        ROOT / "policies/information_grounding_policy.json",
        information_policy=info,
        firewall_policy=firewall,
        freshness_policy=freshness,
    )
    evidence = load_information_research_evidence_policy(
        ROOT / "policies/information_research_evidence_policy.json",
        information_policy=info,
        orchestration_policy=orchestration,
        firewall_policy=firewall,
        freshness_policy=freshness,
        grounding_policy=grounding,
    )
    return info, orchestration, firewall, freshness, grounding, evidence


def evidence_pipeline() -> DeterministicInformationResearchEvidencePipeline:
    return DeterministicInformationResearchEvidencePipeline(*information_policies())


def make_run() -> InformationResearchRun:
    query = InformationQuery.create(
        query_id="query-1", text="launch date evidence", created_at=NOW
    )
    request = InformationResearchRequest(
        request_id="research-request-1",
        query=query,
        operations=("search", "fetch"),
        max_search_calls=1,
        max_fetch_calls=8,
        max_sources=8,
        request_timeout_seconds=10,
        total_timeout_seconds=45,
    )
    request.validate()
    urls = ("https://alpha.example/report", "https://beta.example/report")
    results = tuple(
        InformationSearchResult.create(
            result_id=f"result-{index}",
            query_id=query.query_id,
            provider="search-fixture",
            rank=index,
            title=f"Report {index}",
            url=url,
            snippet="Public evidence",
            retrieved_at=NOW,
        )
        for index, url in enumerate(urls, 1)
    )
    sources = tuple(
        InformationSourceDocument.create(
            source_id=f"source-{index}",
            provider="fetch-fixture",
            url=url,
            title=f"Report {index}",
            normalized_text=CLAIM,
            retrieved_at=NOW,
        )
        for index, url in enumerate(urls, 1)
    )
    activities = (
        InformationActivityRecord(
            activity_id="research-request-1:search:1",
            request_id=request.request_id,
            operation="search",
            provider="search-fixture",
            status="succeeded",
            started_at=NOW,
            finished_at=NOW,
            query_sha256=query.content_sha256,
        ),
        *(
            InformationActivityRecord(
                activity_id=f"research-request-1:fetch:{index}",
                request_id=request.request_id,
                operation="fetch",
                provider="fetch-fixture",
                status="succeeded",
                started_at=NOW,
                finished_at=NOW,
                query_sha256=query.content_sha256,
                source_ids=(source.source_id,),
            )
            for index, source in enumerate(sources, 1)
        ),
    )
    receipt = InformationResearchRunReceipt.create(
        request_id=request.request_id,
        query_id=query.query_id,
        query_sha256=query.content_sha256,
        search_provider="search-fixture",
        fetch_provider="fetch-fixture",
        outcome="completed",
        stopping_reason="all_selected_sources_fetched",
        search_calls=1,
        fetch_calls=2,
        failed_fetch_calls=0,
        selected_result_ids=("result-1", "result-2"),
        failed_result_ids=(),
        source_ids=tuple(source.source_id for source in sources),
        source_content_sha256s=tuple(source.content_sha256 for source in sources),
        started_at=NOW,
        finished_at=NOW,
        policy_version="1.0.0",
        activity_records=activities,
    )
    run = InformationResearchRun(
        request=request,
        search_results=results,
        sources=sources,
        receipt=receipt,
    )
    run.validate(policy=information_policies()[1])
    return run


def make_evidence(pipeline: DeterministicInformationResearchEvidencePipeline):
    run = make_run()
    spans = tuple(
        InformationSupportSpan.create(
            source=source,
            start_character=0,
            end_character=len(CLAIM),
        )
        for source in run.sources
    )
    draft = InformationClaimDraft.create(
        claim_id="claim-1",
        text=CLAIM,
        knowledge_status="verified_fact",
        confidence=0.9,
        support_spans=spans,
    )
    return pipeline.process(
        research_run=run,
        reference_time=NOW,
        outcome="answerable",
        claim_drafts=(draft,),
        created_at=NOW,
    )


def command(*, grounding=None, suffix: str = "1") -> ConversationTurnCommand:
    return ConversationTurnCommand(
        session_id="session-1",
        turn_id=f"turn-{suffix}",
        user_message_id=f"user-{suffix}",
        assistant_message_id=f"assistant-{suffix}",
        request_id=f"request-{suffix}",
        generation_id=f"generation-{suffix}",
        provider="fixture",
        model="fixture-model",
        user_content="What is the launch date?",
        grounding=grounding,
    )


class RecordingRunner:
    def __init__(
        self,
        *,
        invoke_hook: bool = True,
        replayed: bool = False,
        partial_citations: bool = False,
        substituted_grounding: bool = False,
    ) -> None:
        self.invoke_hook = invoke_hook
        self.replayed = replayed
        self.partial_citations = partial_citations
        self.substituted_grounding = substituted_grounding
        self.calls = 0
        self.commands: list[ConversationTurnCommand] = []
        self.hook_calls = 0

    def run_turn(
        self,
        selected: ConversationTurnCommand,
        *,
        cancellation=None,
        response_validation_hook=None,
    ) -> ConversationTurnResult:
        selected.validate()
        self.calls += 1
        self.commands.append(selected)
        if selected.grounding is None or not selected.grounding.claims:
            content = "Local-only response."
        else:
            claim = selected.grounding.claims[0]
            tokens = [item.token for item in claim.citations]
            if self.partial_citations:
                tokens = tokens[:1]
            content = f"{claim.text} {' '.join(tokens)}"
        response = ModelResponse(
            request_id=selected.request_id,
            provider=selected.provider,
            model=selected.model,
            content=content,
            finish_reason="stop",
            created_at=NOW,
        )
        response.validate()
        if response_validation_hook is not None and self.invoke_hook and not self.replayed:
            self.hook_calls += 1
            hook_grounding = None if self.substituted_grounding else selected.grounding
            response_validation_hook(response, hook_grounding)
        packet_id = selected.grounding.packet_id if selected.grounding else None
        packet_sha = (
            conversation_grounding_packet_sha256(selected.grounding)
            if selected.grounding
            else None
        )
        assistant = ConversationMessage.create(
            message_id=selected.assistant_message_id,
            turn_id=selected.turn_id,
            role="assistant",
            content=content,
            created_at=NOW,
            data_classification=selected.data_classification,
        )
        result = ConversationTurnResult(
            session_id=selected.session_id,
            turn_id=selected.turn_id,
            request_id=selected.request_id,
            generation_id=selected.generation_id,
            provider=selected.provider,
            model=selected.model,
            assistant_message=assistant,
            response=response,
            grounding_packet_id=packet_id,
            grounding_packet_sha256=packet_sha,
            validation_outcome="accepted",
            replayed=self.replayed,
        )
        result.validate()
        return result


def adapter(runner: RecordingRunner):
    pipeline = evidence_pipeline()
    response_policy = load_conversation_response_validation_policy(
        ROOT / "policies/conversation_response_validation_policy.json"
    )
    bridge_policy = load_information_conversation_bridge_policy(
        ROOT / "policies/information_conversation_bridge_policy.json",
        grounding_policy=pipeline.grounding_policy,
        response_validation_policy=response_policy,
    )
    mode_policy = load_information_research_mode_policy(
        ROOT / "policies/information_research_mode_policy.json",
        evidence_policy=pipeline.evidence_policy,
        bridge_policy=bridge_policy,
        response_validation_policy=response_policy,
    )
    selected = DeterministicInformationResearchModeAdapter(
        orchestrator=runner,
        evidence_pipeline=pipeline,
        bridge_policy=bridge_policy,
        response_validation_policy=response_policy,
        mode_policy=mode_policy,
        clock=lambda: NOW,
    )
    return selected, pipeline


def memory_grounding() -> ConversationGroundingPacket:
    citation = ConversationCitation(
        citation_id="memory-citation-1",
        source_kind="memory_source",
        source_ref="memory-1",
        token="[memory:memory-1]",
        data_classification="PRIVATE",
    )
    claim = ConversationGroundingClaim(
        claim_id="memory-claim-1",
        text="The saved preference is concise answers.",
        content_sha256=sha256_text("The saved preference is concise answers."),
        knowledge_status="verified_fact",
        confidence=0.9,
        data_classification="PRIVATE",
        citations=(citation,),
    )
    packet = ConversationGroundingPacket(
        packet_id="memory-packet-1",
        outcome="answerable",
        claims=(claim,),
        created_at=NOW,
        max_classification="PRIVATE",
    )
    packet.validate()
    return packet


def test_local_only_turn_has_no_web_artifacts() -> None:
    runner = RecordingRunner()
    selected, _ = adapter(runner)
    result = selected.run_turn(
        command(), mode="local_only", availability="not_requested"
    )
    result.validate(adapter=selected, command=command(), evidence_result=None)
    assert runner.calls == 1
    assert runner.hook_calls == 0
    assert result.status == "completed"
    assert result.source_summaries == ()
    assert result.receipt.research_run_id is None


def test_local_only_allows_existing_private_grounding() -> None:
    runner = RecordingRunner()
    selected, _ = adapter(runner)
    cmd = command(grounding=memory_grounding())
    result = selected.run_turn(cmd, mode="local_only", availability="not_requested")
    assert result.receipt.conversation_packet_id == cmd.grounding.packet_id
    assert result.receipt.source_summaries == ()


@pytest.mark.parametrize(
    "availability,has_evidence",
    [("available", False), ("not_requested", True)],
)
def test_local_only_rejects_research_state(availability: str, has_evidence: bool) -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    evidence = make_evidence(pipeline) if has_evidence else None
    with pytest.raises(InformationResearchModeError):
        selected.run_turn(
            command(),
            mode="local_only",
            availability=availability,
            evidence_result=evidence,
        )
    assert runner.calls == 0


def test_local_only_rejects_web_grounding() -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    evidence = make_evidence(pipeline)
    from alice_information.conversation_bridge import project_information_grounding_to_conversation

    projection = project_information_grounding_to_conversation(
        verified_grounding=evidence.grounding,
        query=evidence.research_run.request.query,
        qualified_sources=evidence.qualified_sources,
        information_policy=pipeline.information_policy,
        firewall_policy=pipeline.firewall_policy,
        freshness_policy=pipeline.freshness_policy,
        grounding_policy=pipeline.grounding_policy,
        bridge_policy=selected.bridge_policy,
    )
    with pytest.raises(InformationResearchModeError):
        selected.run_turn(
            command(grounding=projection.conversation_packet),
            mode="local_only",
            availability="not_requested",
        )
    assert runner.calls == 0


@pytest.mark.parametrize("availability", ["offline", "unavailable"])
def test_unavailable_research_returns_before_runner_call(availability: str) -> None:
    runner = RecordingRunner()
    selected, _ = adapter(runner)
    result = selected.run_turn(
        command(), mode="research", availability=availability
    )
    result.validate(adapter=selected, command=command(), evidence_result=None)
    assert result.status == "unavailable"
    assert result.conversation_result is None
    assert runner.calls == 0
    assert result.receipt.unavailable_reason == availability


def test_unavailable_research_rejects_supplied_evidence() -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    with pytest.raises(InformationResearchModeError):
        selected.run_turn(
            command(),
            mode="research",
            availability="offline",
            evidence_result=make_evidence(pipeline),
        )
    assert runner.calls == 0


def test_available_research_requires_evidence_and_clean_command() -> None:
    runner = RecordingRunner()
    selected, _ = adapter(runner)
    with pytest.raises(InformationResearchModeError):
        selected.run_turn(command(), mode="research", availability="available")
    with pytest.raises(InformationResearchModeError):
        selected.run_turn(
            command(grounding=memory_grounding()),
            mode="research",
            availability="available",
        )
    assert runner.calls == 0


def test_available_research_projects_and_validates_before_commit() -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    evidence = make_evidence(pipeline)
    cmd = command()
    result = selected.run_turn(
        cmd,
        mode="research",
        availability="available",
        evidence_result=evidence,
    )
    result.validate(adapter=selected, command=cmd, evidence_result=evidence)
    assert runner.calls == 1
    assert runner.hook_calls == 1
    assert runner.commands[0].grounding == result.projection.conversation_packet
    assert len(result.source_summaries) == 2
    assert all(item.citation_token.startswith("[WEB:") for item in result.source_summaries)
    metadata = result.receipt.to_metadata_record()
    assert CLAIM not in repr(metadata)
    assert "normalized_text" not in repr(metadata)


def test_replayed_research_response_is_postvalidated() -> None:
    runner = RecordingRunner(replayed=True)
    selected, pipeline = adapter(runner)
    evidence = make_evidence(pipeline)
    result = selected.run_turn(
        command(),
        mode="research",
        availability="available",
        evidence_result=evidence,
    )
    assert result.conversation_result.replayed is True
    assert runner.hook_calls == 0
    assert result.response_validation is not None


def test_runner_must_invoke_precommit_hook() -> None:
    runner = RecordingRunner(invoke_hook=False)
    selected, pipeline = adapter(runner)
    with pytest.raises(InformationResearchModeError) as raised:
        selected.run_turn(
            command(),
            mode="research",
            availability="available",
            evidence_result=make_evidence(pipeline),
        )
    assert raised.value.code == "research_response_validation_missing"


def test_substituted_grounding_is_rejected_by_hook() -> None:
    runner = RecordingRunner(substituted_grounding=True)
    selected, pipeline = adapter(runner)
    with pytest.raises(Exception):
        selected.run_turn(
            command(),
            mode="research",
            availability="available",
            evidence_result=make_evidence(pipeline),
        )


def test_partial_web_citation_set_is_rejected_before_result() -> None:
    runner = RecordingRunner(partial_citations=True)
    selected, pipeline = adapter(runner)
    with pytest.raises(Exception):
        selected.run_turn(
            command(),
            mode="research",
            availability="available",
            evidence_result=make_evidence(pipeline),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("research_receipt_sha256", "0" * 64),
        ("evidence_receipt_sha256", "0" * 64),
        ("projection_sha256", "0" * 64),
        ("validation_sha256", "0" * 64),
        ("response_sha256", "0" * 64),
    ],
)
def test_receipt_rejects_binding_tampering(field: str, value: object) -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    result = selected.run_turn(
        command(),
        mode="research",
        availability="available",
        evidence_result=make_evidence(pipeline),
    )
    forged = replace(result.receipt, **{field: value})
    with pytest.raises(InformationResearchModeError):
        forged.validate()


def test_source_summary_substitution_is_rejected() -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    evidence = make_evidence(pipeline)
    result = selected.run_turn(
        command(),
        mode="research",
        availability="available",
        evidence_result=evidence,
    )
    changed = replace(result.source_summaries[0], canonical_url="https://evil.example/")
    forged = replace(result, source_summaries=(changed, *result.source_summaries[1:]))
    with pytest.raises(InformationResearchModeError):
        forged.validate(
            adapter=selected,
            command=command(),
            evidence_result=evidence,
        )



def test_result_rejects_policy_version_and_command_binding_substitution() -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    evidence = make_evidence(pipeline)
    cmd = command()
    result = selected.run_turn(
        cmd,
        mode="research",
        availability="available",
        evidence_result=evidence,
    )
    changed_receipt = replace(
        result.receipt,
        policy_versions=("wrong@1.0.0", *result.receipt.policy_versions[1:]),
    )
    forged = replace(result, receipt=changed_receipt)
    with pytest.raises(InformationResearchModeError):
        forged.validate(adapter=selected, command=cmd, evidence_result=evidence)
    with pytest.raises(InformationResearchModeError):
        result.validate(
            adapter=selected,
            command=replace(cmd, request_id="request-other"),
            evidence_result=evidence,
        )


def test_source_summary_rejects_noncanonical_url() -> None:
    runner = RecordingRunner()
    selected, pipeline = adapter(runner)
    result = selected.run_turn(
        command(),
        mode="research",
        availability="available",
        evidence_result=make_evidence(pipeline),
    )
    changed = replace(result.source_summaries[0], canonical_url="http://user:pass@evil.example/#x")
    with pytest.raises(InformationResearchModeError):
        changed.validate()

def test_unknown_mode_and_availability_fail_before_runner() -> None:
    runner = RecordingRunner()
    selected, _ = adapter(runner)
    with pytest.raises(InformationResearchModeError):
        selected.run_turn(command(), mode="silent", availability="not_requested")
    with pytest.raises(InformationResearchModeError):
        selected.run_turn(command(), mode="research", availability="maybe")
    assert runner.calls == 0
