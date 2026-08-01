from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from alice_information.live_acceptance import InformationLiveAcceptancePolicy, InformationLiveAcceptanceRecord
from alice_information.live_provider_contracts import canonical_sha256
from alice_information.live_research import InformationLiveResearchReceipt, InformationLiveSourceOutcome

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "policies/information_live_acceptance_release_policy.json"
BENCHMARK_PATH = ROOT / "benchmarks/phase4/information_live_acceptance_v1.json"
D = "d" * 64
COMMIT = "a" * 40
ROLLBACK = "b" * 40
NOW = "2026-07-30T12:00:00Z"
BOUNDARIES = MappingProxyType({
    "public_queries_only": True,
    "foreground_only": True,
    "fallback_allowed": False,
    "retry_allowed": False,
    "recursive_browsing_allowed": False,
    "source_body_persistence_allowed": False,
    "phase5_storage_allowed": False,
    "memory_write_allowed": False,
    "external_action_allowed": False,
    "background_execution_allowed": False,
    "private_record_only": True,
})


def policy() -> InformationLiveAcceptancePolicy:
    return InformationLiveAcceptancePolicy.load(POLICY_PATH)


def source_outcome() -> InformationLiveSourceOutcome:
    value = InformationLiveSourceOutcome(
        source_id="source-1",
        canonical_url="https://example.com/public",
        source_content_sha256=D,
        temporal_verdict="accepted",
        inspection_verdict="clear",
        freshness_verdict="fresh",
        supports_claim=True,
        disposition="grounded",
        reason_code=None,
    )
    value.validate()
    return value


def live_receipt() -> InformationLiveResearchReceipt:
    outcome_record = source_outcome().metadata_record()
    return InformationLiveResearchReceipt.create(
        policy_version="1.0.0",
        request_id="request-1",
        query_id="query-1",
        query_sha256=D,
        outcome="answerable",
        search_result_count=1,
        search_receipt_sha256=D,
        fetch_attempt_count=1,
        fetch_attempt_sequence_sha256=D,
        fetch_receipt_sha256s=(D,),
        fetch_failure_sha256s=(),
        temporal_resolution_sha256s=(D,),
        source_outcome_sha256=canonical_sha256([outcome_record]),
        grounded_source_sha256s=(D,),
        grounding_sha256=D,
        projection_sha256=D,
        conversation_packet_sha256=D,
        response_sha256=D,
        validation_sha256=D,
        citation_validation_outcome="accepted",
        pre_commit_validation_count=1,
        policy_bindings=("p410a@1.0.0", "p410b@1.0.0"),
        created_at=NOW,
    )


def record(**overrides) -> InformationLiveAcceptanceRecord:
    p = policy()
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    values = {
        "approved": True,
        "audit_version": "p4.10c-v1",
        "repository_commit": COMMIT,
        "repository_head_commit": COMMIT,
        "rollback_commit": ROLLBACK,
        "repository_clean": True,
        "repository_snapshot_before_sha256": D,
        "repository_snapshot_after_sha256": D,
        "package_version": "0.18.0",
        "evaluated_at": NOW,
        "policy_id": p.policy_name,
        "policy_sha256": p.policy_sha256,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_sha256": canonical_sha256(benchmark),
        "provider_policy_binding": "alice_information_live_provider_runtime_policy@1.0.0:" + D,
        "research_policy_binding": "alice_information_live_research_policy@1.0.0:" + D,
        "deterministic_test_collected": 101,
        "deterministic_test_passed": 101,
        "deterministic_test_skipped": 0,
        "deterministic_test_output_sha256": D,
        "repository_regression_collected": 2126,
        "repository_regression_passed": 2124,
        "repository_regression_skipped": 2,
        "repository_regression_subtests_passed": 14,
        "repository_regression_output_sha256": D,
        "live_research_receipt": MappingProxyType(live_receipt().to_metadata_record()),
        "source_outcomes": (MappingProxyType(source_outcome().metadata_record()),),
        "acceptance_domains": p.required_acceptance_domains,
        "decision_reasons": (),
        "boundaries": BOUNDARIES,
    }
    values.update(overrides)
    return InformationLiveAcceptanceRecord.create(**values)
