"""Authoritative-memory and source-record citation evaluation for P2.9b."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

from .cited_answer import (
    MemoryAnswerAuthorization,
    MemoryAnswerClaim,
    MemoryAnswerCitation,
    MemoryAnswerSubmission,
    MemoryCitedAnswerAuthorizationError,
    ORDINARY_ANSWER_CLASSIFICATIONS,
)
from .evaluation_contract import (
    MemoryEvaluationBenchmark,
    MemoryEvaluationCase,
    MemoryEvaluationPolicy,
    load_memory_evaluation_policy,
)
from .service import (
    MemoryContentAccessAuthorization,
    MemoryNotFoundError,
    load_memory,
    load_memory_content,
)

_CLASSIFICATION_RANK = {
    value: index
    for index, value in enumerate(ORDINARY_ANSWER_CLASSIFICATIONS)
}

_SUPPORTING_RELATIONS = {
    "supports",
    "derived_from",
}

_SOURCE_REQUIRED_OUTCOMES = {
    "answerable",
    "conflict",
    "uncertain",
}


class MemoryCitationEvaluationError(RuntimeError):
    """Raised when citation evaluation cannot be performed safely."""


@dataclass(frozen=True)
class MemoryAnswerCaseEvaluation:
    case_id: str
    suite: str
    expected_outcome: str
    actual_outcome: str
    passed: bool
    issues: tuple[str, ...]
    actual_memory_ids: tuple[str, ...]
    actual_source_refs: tuple[str, ...]
    actual_knowledge_statuses: tuple[str, ...]
    claim_count: int
    supported_claim_count: int
    cited_claim_count: int
    unsupported_claim_count: int
    expected_sources_present: bool
    forbidden_memory_hits: tuple[str, ...]
    forbidden_candidate_hits: tuple[str, ...]


@dataclass(frozen=True)
class MemoryCitationEvaluationSummary:
    benchmark_id: str
    policy_id: str
    case_count: int
    passed_case_count: int
    case_pass_rate: float
    claim_count: int
    supported_claim_count: int
    cited_claim_count: int
    unsupported_claim_count: int
    personal_source_attribution_rate: float
    expected_source_citation_rate: float
    claim_citation_coverage: float
    unsupported_personal_claim_rate: float
    passes_source_attribution_gate: bool
    passes_unsupported_claim_gate: bool
    passes_claim_citation_gate: bool
    passes_all_p29b_gates: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    private_output_only: bool
    cases: tuple[MemoryAnswerCaseEvaluation, ...]


def _require_authorization(
    authorization: MemoryAnswerAuthorization,
) -> None:
    if not authorization.allowed:
        raise MemoryCitedAnswerAuthorizationError(
            "Citation evaluation denied by explicit authorization."
        )
    if not authorization.actor.strip() or not authorization.purpose.strip():
        raise MemoryCitedAnswerAuthorizationError(
            "Citation evaluation requires actor and purpose."
        )
    if authorization.max_classification not in _CLASSIFICATION_RANK:
        raise MemoryCitedAnswerAuthorizationError(
            "Citation evaluation cannot authorize HIGHLY_SENSITIVE data."
        )


def _citation_row(
    connection: sqlite3.Connection,
    *,
    citation: MemoryAnswerCitation,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT
            memory_source_id,
            memory_id,
            source_type,
            source_ref,
            source_content_sha256,
            source_text_sha256,
            chunk_id,
            file_id,
            support_relation
        FROM memory_sources
        WHERE memory_source_id = ?
        """,
        (citation.memory_source_id,),
    ).fetchone()


def _citation_matches_row(
    citation: MemoryAnswerCitation,
    row: sqlite3.Row,
) -> bool:
    fields = (
        "memory_source_id",
        "memory_id",
        "source_type",
        "source_ref",
        "source_content_sha256",
        "source_text_sha256",
        "chunk_id",
        "file_id",
        "support_relation",
    )
    return all(
        getattr(citation, field) == row[field]
        for field in fields
    )


def _verify_claim(
    connection: sqlite3.Connection,
    *,
    claim: MemoryAnswerClaim,
    submission: MemoryAnswerSubmission,
    authorization: MemoryAnswerAuthorization,
) -> tuple[bool, bool, tuple[str, ...], tuple[str, ...]]:
    issues: list[str] = []
    source_refs: list[str] = []

    try:
        record = load_memory(connection, memory_id=claim.memory_id)
    except MemoryNotFoundError:
        return (
            False,
            False,
            ("claim_memory_missing",),
            (),
        )

    rank = _CLASSIFICATION_RANK.get(record.data_classification)
    if rank is None:
        issues.append("claim_uses_highly_sensitive_memory")
    elif rank > _CLASSIFICATION_RANK[authorization.max_classification]:
        issues.append("claim_exceeds_authorization_classification")

    if record.deletion_state != "active":
        issues.append("claim_memory_not_active")

    content_authorization = MemoryContentAccessAuthorization(
        actor=authorization.actor,
        allowed=True,
        reason=authorization.purpose,
    )
    try:
        content = load_memory_content(
            connection,
            memory_id=claim.memory_id,
            authorization=content_authorization,
        )
    except Exception:
        issues.append("claim_content_not_authorized")
        content = None

    if content is not None:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if claim.text != content:
            issues.append("claim_text_not_authoritative")
        if claim.content_sha256 != digest:
            issues.append("claim_digest_not_authoritative")
        if record.content_sha256 != digest:
            issues.append("stored_content_digest_mismatch")

    if claim.knowledge_status != record.knowledge_status:
        issues.append("claim_knowledge_status_mismatch")
    if claim.confidence != record.confidence:
        issues.append("claim_confidence_mismatch")
    if claim.data_classification != record.data_classification:
        issues.append("claim_classification_mismatch")

    expected_claim_id = hashlib.sha256(
        (
            f"{submission.case_id}|{claim.memory_id}|"
            f"{claim.content_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    if claim.claim_id != expected_claim_id:
        issues.append("claim_id_mismatch")

    valid_citations = 0
    if not claim.citations:
        issues.append("claim_has_no_citation")
    if len({item.memory_source_id for item in claim.citations}) != len(
        claim.citations
    ):
        issues.append("claim_has_duplicate_citation")

    for citation in claim.citations:
        if citation.memory_id != claim.memory_id:
            issues.append("citation_memory_mismatch")
            continue
        row = _citation_row(connection, citation=citation)
        if row is None:
            issues.append("citation_source_row_missing")
            continue
        if not _citation_matches_row(citation, row):
            issues.append("citation_source_row_mismatch")
            continue
        if citation.support_relation not in _SUPPORTING_RELATIONS:
            issues.append("citation_is_not_supporting")
            continue
        valid_citations += 1
        source_refs.append(citation.source_ref)

    cited_in_answer = bool(claim.citations) and all(
        citation.token in submission.answer_text
        for citation in claim.citations
    )
    if not cited_in_answer:
        issues.append("citation_token_missing_from_answer")

    supported = (
        not issues
        and valid_citations == len(claim.citations)
        and valid_citations > 0
    )
    return supported, cited_in_answer, tuple(sorted(set(issues))), tuple(source_refs)


def evaluate_memory_answer_submission(
    connection: sqlite3.Connection,
    *,
    case: MemoryEvaluationCase,
    submission: MemoryAnswerSubmission,
    authorization: MemoryAnswerAuthorization,
) -> MemoryAnswerCaseEvaluation:
    """Validate one answer against authoritative memory and benchmark intent."""
    _require_authorization(authorization)
    issues: list[str] = []

    if submission.case_id != case.case_id:
        issues.append("case_id_mismatch")
    if submission.outcome != case.expected_outcome:
        issues.append("outcome_mismatch")

    claim_ids = tuple(claim.claim_id for claim in submission.claims)
    memory_ids = tuple(claim.memory_id for claim in submission.claims)
    if len(set(claim_ids)) != len(claim_ids):
        issues.append("duplicate_claim_id")
    if len(set(memory_ids)) != len(memory_ids):
        issues.append("duplicate_claim_memory")

    if case.expected_outcome not in _SOURCE_REQUIRED_OUTCOMES:
        if submission.claims:
            issues.append("nonanswer_outcome_contains_claims")
    elif not submission.claims:
        issues.append("answer_outcome_has_no_claims")

    supported_claim_count = 0
    cited_claim_count = 0
    claim_issue_count = 0
    source_refs: list[str] = []
    knowledge_statuses: list[str] = []

    for claim in submission.claims:
        supported, cited, claim_issues, claim_sources = _verify_claim(
            connection,
            claim=claim,
            submission=submission,
            authorization=authorization,
        )
        if supported:
            supported_claim_count += 1
        if cited:
            cited_claim_count += 1
        if claim_issues:
            claim_issue_count += 1
            issues.extend(claim_issues)
        source_refs.extend(claim_sources)
        knowledge_statuses.append(claim.knowledge_status)

    actual_memory_set = set(memory_ids)
    actual_source_set = set(source_refs)
    expected_memory_set = set(case.expected_memory_ids)
    expected_source_set = set(case.expected_source_refs)

    if actual_memory_set != expected_memory_set:
        issues.append("expected_memory_set_mismatch")
    expected_sources_present = expected_source_set <= actual_source_set
    if actual_source_set != expected_source_set:
        issues.append("expected_source_set_mismatch")
    if set(knowledge_statuses) != set(case.expected_knowledge_statuses):
        issues.append("expected_knowledge_status_set_mismatch")

    forbidden_memory_hits = tuple(
        sorted(actual_memory_set.intersection(case.forbidden_memory_ids))
    )
    if forbidden_memory_hits:
        issues.append("forbidden_memory_exposed")

    forbidden_candidate_hits = tuple(
        sorted(
            candidate_id
            for candidate_id in case.forbidden_candidate_ids
            if candidate_id in submission.answer_text
        )
    )
    if forbidden_candidate_hits:
        issues.append("forbidden_candidate_exposed")

    unique_issues = tuple(sorted(set(issues)))
    return MemoryAnswerCaseEvaluation(
        case_id=case.case_id,
        suite=case.suite,
        expected_outcome=case.expected_outcome,
        actual_outcome=submission.outcome,
        passed=not unique_issues,
        issues=unique_issues,
        actual_memory_ids=tuple(sorted(actual_memory_set)),
        actual_source_refs=tuple(sorted(actual_source_set)),
        actual_knowledge_statuses=tuple(sorted(set(knowledge_statuses))),
        claim_count=len(submission.claims),
        supported_claim_count=supported_claim_count,
        cited_claim_count=cited_claim_count,
        unsupported_claim_count=claim_issue_count,
        expected_sources_present=expected_sources_present,
        forbidden_memory_hits=forbidden_memory_hits,
        forbidden_candidate_hits=forbidden_candidate_hits,
    )


def _metric_gate(
    policy: MemoryEvaluationPolicy,
    metric_id: str,
) -> tuple[str, float]:
    for gate in policy.metric_gates:
        if gate.metric_id == metric_id:
            return gate.direction, gate.threshold
    raise MemoryCitationEvaluationError(
        f"Evaluation policy lacks metric gate: {metric_id}"
    )


def _passes(value: float, gate: tuple[str, float]) -> bool:
    direction, threshold = gate
    if direction == "minimum":
        return value >= threshold
    return value <= threshold


def evaluate_memory_answer_submissions(
    connection: sqlite3.Connection,
    *,
    benchmark: MemoryEvaluationBenchmark,
    submissions: tuple[MemoryAnswerSubmission, ...],
    authorization: MemoryAnswerAuthorization,
    policy: MemoryEvaluationPolicy | None = None,
) -> MemoryCitationEvaluationSummary:
    """Evaluate exactly one source-cited submission for every benchmark case."""
    _require_authorization(authorization)
    resolved_policy = policy or load_memory_evaluation_policy()

    by_case = {submission.case_id: submission for submission in submissions}
    if len(by_case) != len(submissions):
        raise MemoryCitationEvaluationError(
            "Citation evaluation contains duplicate case submissions."
        )
    expected_case_ids = {case.case_id for case in benchmark.cases}
    actual_case_ids = set(by_case)
    if actual_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - actual_case_ids)
        extra = sorted(actual_case_ids - expected_case_ids)
        raise MemoryCitationEvaluationError(
            "Citation evaluation case coverage mismatch; missing="
            f"{missing}, extra={extra}"
        )

    evaluations = tuple(
        evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=by_case[case.case_id],
            authorization=authorization,
        )
        for case in benchmark.cases
    )

    claim_count = sum(item.claim_count for item in evaluations)
    supported_claim_count = sum(
        item.supported_claim_count for item in evaluations
    )
    cited_claim_count = sum(item.cited_claim_count for item in evaluations)
    unsupported_claim_count = sum(
        item.unsupported_claim_count for item in evaluations
    )
    source_required = tuple(
        item
        for item in evaluations
        if item.expected_outcome in _SOURCE_REQUIRED_OUTCOMES
    )

    source_rate = (
        supported_claim_count / claim_count
        if claim_count
        else 1.0
    )
    claim_coverage = (
        cited_claim_count / claim_count
        if claim_count
        else 1.0
    )
    unsupported_rate = (
        unsupported_claim_count / claim_count
        if claim_count
        else 0.0
    )
    expected_source_rate = (
        sum(item.expected_sources_present for item in source_required)
        / len(source_required)
        if source_required
        else 1.0
    )

    passes_source = _passes(
        source_rate,
        _metric_gate(resolved_policy, "personal_source_attribution_rate"),
    )
    passes_unsupported = _passes(
        unsupported_rate,
        _metric_gate(resolved_policy, "unsupported_personal_claim_rate"),
    )
    passes_claim_citation = claim_coverage == 1.0
    passed_cases = sum(item.passed for item in evaluations)

    return MemoryCitationEvaluationSummary(
        benchmark_id=benchmark.benchmark_id,
        policy_id=resolved_policy.policy_id,
        case_count=len(evaluations),
        passed_case_count=passed_cases,
        case_pass_rate=(
            passed_cases / len(evaluations)
            if evaluations
            else 1.0
        ),
        claim_count=claim_count,
        supported_claim_count=supported_claim_count,
        cited_claim_count=cited_claim_count,
        unsupported_claim_count=unsupported_claim_count,
        personal_source_attribution_rate=round(source_rate, 6),
        expected_source_citation_rate=round(expected_source_rate, 6),
        claim_citation_coverage=round(claim_coverage, 6),
        unsupported_personal_claim_rate=round(unsupported_rate, 6),
        passes_source_attribution_gate=passes_source,
        passes_unsupported_claim_gate=passes_unsupported,
        passes_claim_citation_gate=passes_claim_citation,
        passes_all_p29b_gates=(
            passes_source
            and passes_unsupported
            and passes_claim_citation
            and passed_cases == len(evaluations)
        ),
        memory_write_allowed=resolved_policy.memory_write_allowed,
        external_action_allowed=resolved_policy.external_action_allowed,
        tool_calling_allowed=resolved_policy.tool_calling_allowed,
        web_access_allowed=resolved_policy.web_access_allowed,
        private_output_only=resolved_policy.private_output_only,
        cases=evaluations,
    )
