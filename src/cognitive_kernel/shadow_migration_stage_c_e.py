"""Read-only destination-candidate and shadow-read evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import ProductHostScope

SHADOW_MIGRATION_STAGE_C_E_SCHEMA_VERSION = "1.0.0"
SHADOW_WORKLOAD_CLASSES = frozenset({"synthetic", "owner_authorized"})
SHADOW_COMPARISON_STATES = frozenset(
    {"equivalent", "candidate_improved", "candidate_degraded", "inconclusive"}
)
DESTINATION_RECOMMENDATIONS = frozenset(
    {"continue_shadow_evaluation", "eligible_for_next_research_gate", "repair_before_continuation"}
)


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise CognitiveKernelContractError(f"{field} must be boolean")
    return value


def _require_non_negative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveKernelContractError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise CognitiveKernelContractError(f"{field} must be finite and non-negative")
    return normalized


def _require_choice(value: object, field: str, allowed: frozenset[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not an allowed value")
    return normalized


@dataclass(frozen=True)
class DestinationCandidateProfile:
    scope: ProductHostScope
    candidate_id: str
    schema_version: str
    backend_types: tuple[str, ...]
    component_ids: tuple[str, ...]
    contract_roles: tuple[str, ...]
    deployment_profiles: tuple[str, ...]
    prototype_state: str
    created_at: str
    production_authority: bool
    production_influence: bool
    profile_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        candidate_id: object,
        backend_types: tuple[object, ...] | list[object],
        component_ids: tuple[object, ...] | list[object],
        contract_roles: tuple[object, ...] | list[object],
        deployment_profiles: tuple[object, ...] | list[object],
        created_at: object,
        prototype_state: object = "research_candidate",
        production_authority: object = False,
        production_influence: object = False,
        schema_version: object = SHADOW_MIGRATION_STAGE_C_E_SCHEMA_VERSION,
    ) -> "DestinationCandidateProfile":
        scope.validate()
        record = cls(
            scope=scope,
            candidate_id=require_identifier(candidate_id, "candidate_id"),
            schema_version=require_schema_version(schema_version),
            backend_types=normalize_identifier_sequence(backend_types, "backend_types"),
            component_ids=normalize_identifier_sequence(component_ids, "component_ids"),
            contract_roles=normalize_identifier_sequence(contract_roles, "contract_roles"),
            deployment_profiles=normalize_identifier_sequence(deployment_profiles, "deployment_profiles"),
            prototype_state=require_identifier(prototype_state, "prototype_state"),
            created_at=normalize_timestamp(created_at, "created_at"),
            production_authority=_require_bool(production_authority, "production_authority"),
            production_influence=_require_bool(production_influence, "production_influence"),
            profile_sha256="0" * 64,
        )
        if record.production_authority or record.production_influence:
            raise CognitiveKernelContractError(
                "Stage C+E candidate profiles may not activate production authority or influence"
            )
        if not record.backend_types or not record.component_ids or not record.contract_roles:
            raise CognitiveKernelContractError("candidate capability descriptors may not be empty")
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "profile_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope.metadata_record(),
            "candidate_id": self.candidate_id,
            "schema_version": self.schema_version,
            "backend_types": list(self.backend_types),
            "component_ids": list(self.component_ids),
            "contract_roles": list(self.contract_roles),
            "deployment_profiles": list(self.deployment_profiles),
            "prototype_state": self.prototype_state,
            "created_at": self.created_at,
            "production_authority": self.production_authority,
            "production_influence": self.production_influence,
        }
        if include_digest:
            payload["profile_sha256"] = self.profile_sha256
        return payload

    def validate(self) -> None:
        recreated = DestinationCandidateProfile.create(
            scope=self.scope,
            candidate_id=self.candidate_id,
            backend_types=self.backend_types,
            component_ids=self.component_ids,
            contract_roles=self.contract_roles,
            deployment_profiles=self.deployment_profiles,
            created_at=self.created_at,
            prototype_state=self.prototype_state,
            production_authority=self.production_authority,
            production_influence=self.production_influence,
            schema_version=self.schema_version,
        )
        if recreated.profile_sha256 != require_sha256(self.profile_sha256, "profile_sha256"):
            raise CognitiveKernelContractError("destination candidate profile digest mismatch")


@dataclass(frozen=True)
class ShadowReadWorkload:
    scope: ProductHostScope
    workload_id: str
    workload_class: str
    query_sha256: str
    expected_record_ids: tuple[str, ...]
    expected_conflict_record_ids: tuple[str, ...]
    expected_correction_record_ids: tuple[str, ...]
    expected_deleted_record_ids: tuple[str, ...]
    authorization_reference_id: str | None
    created_at: str
    workload_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        workload_id: object,
        workload_class: object,
        query_sha256: object,
        expected_record_ids: tuple[object, ...] | list[object],
        expected_conflict_record_ids: tuple[object, ...] | list[object] = (),
        expected_correction_record_ids: tuple[object, ...] | list[object] = (),
        expected_deleted_record_ids: tuple[object, ...] | list[object] = (),
        authorization_reference_id: object | None = None,
        created_at: object,
    ) -> "ShadowReadWorkload":
        scope.validate()
        workload_class_value = _require_choice(
            workload_class, "workload_class", SHADOW_WORKLOAD_CLASSES
        )
        authorization = (
            require_identifier(authorization_reference_id, "authorization_reference_id")
            if authorization_reference_id is not None
            else None
        )
        if workload_class_value == "owner_authorized" and authorization is None:
            raise CognitiveKernelContractError(
                "owner-authorized workloads require an authorization reference"
            )
        if workload_class_value == "synthetic" and authorization is not None:
            raise CognitiveKernelContractError(
                "synthetic workloads may not claim an owner authorization reference"
            )
        record = cls(
            scope=scope,
            workload_id=require_identifier(workload_id, "workload_id"),
            workload_class=workload_class_value,
            query_sha256=require_sha256(query_sha256, "query_sha256"),
            expected_record_ids=normalize_identifier_sequence(expected_record_ids, "expected_record_ids"),
            expected_conflict_record_ids=normalize_identifier_sequence(
                expected_conflict_record_ids, "expected_conflict_record_ids"
            ),
            expected_correction_record_ids=normalize_identifier_sequence(
                expected_correction_record_ids, "expected_correction_record_ids"
            ),
            expected_deleted_record_ids=normalize_identifier_sequence(
                expected_deleted_record_ids, "expected_deleted_record_ids"
            ),
            authorization_reference_id=authorization,
            created_at=normalize_timestamp(created_at, "created_at"),
            workload_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "workload_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope.metadata_record(),
            "workload_id": self.workload_id,
            "workload_class": self.workload_class,
            "query_sha256": self.query_sha256,
            "expected_record_ids": list(self.expected_record_ids),
            "expected_conflict_record_ids": list(self.expected_conflict_record_ids),
            "expected_correction_record_ids": list(self.expected_correction_record_ids),
            "expected_deleted_record_ids": list(self.expected_deleted_record_ids),
            "authorization_reference_id": self.authorization_reference_id,
            "created_at": self.created_at,
        }
        if include_digest:
            payload["workload_sha256"] = self.workload_sha256
        return payload

    def validate(self) -> None:
        recreated = ShadowReadWorkload.create(
            scope=self.scope,
            workload_id=self.workload_id,
            workload_class=self.workload_class,
            query_sha256=self.query_sha256,
            expected_record_ids=self.expected_record_ids,
            expected_conflict_record_ids=self.expected_conflict_record_ids,
            expected_correction_record_ids=self.expected_correction_record_ids,
            expected_deleted_record_ids=self.expected_deleted_record_ids,
            authorization_reference_id=self.authorization_reference_id,
            created_at=self.created_at,
        )
        if recreated.workload_sha256 != require_sha256(self.workload_sha256, "workload_sha256"):
            raise CognitiveKernelContractError("shadow-read workload digest mismatch")


@dataclass(frozen=True)
class ShadowReadObservation:
    scope: ProductHostScope
    workload_id: str
    candidate_id: str
    result_record_ids: tuple[str, ...]
    conflict_record_ids: tuple[str, ...]
    correction_record_ids: tuple[str, ...]
    deleted_record_ids_returned: tuple[str, ...]
    latency_ms: float
    staleness_ms: float
    product_isolation_passed: bool
    private_payload_exposed: bool
    explanation_trace_sha256: str
    observed_at: str
    observation_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        workload_id: object,
        candidate_id: object,
        result_record_ids: tuple[object, ...] | list[object],
        conflict_record_ids: tuple[object, ...] | list[object] = (),
        correction_record_ids: tuple[object, ...] | list[object] = (),
        deleted_record_ids_returned: tuple[object, ...] | list[object] = (),
        latency_ms: object,
        staleness_ms: object,
        product_isolation_passed: object,
        private_payload_exposed: object,
        explanation_trace_sha256: object,
        observed_at: object,
    ) -> "ShadowReadObservation":
        scope.validate()
        record = cls(
            scope=scope,
            workload_id=require_identifier(workload_id, "workload_id"),
            candidate_id=require_identifier(candidate_id, "candidate_id"),
            result_record_ids=normalize_identifier_sequence(result_record_ids, "result_record_ids"),
            conflict_record_ids=normalize_identifier_sequence(conflict_record_ids, "conflict_record_ids"),
            correction_record_ids=normalize_identifier_sequence(correction_record_ids, "correction_record_ids"),
            deleted_record_ids_returned=normalize_identifier_sequence(
                deleted_record_ids_returned, "deleted_record_ids_returned"
            ),
            latency_ms=_require_non_negative_number(latency_ms, "latency_ms"),
            staleness_ms=_require_non_negative_number(staleness_ms, "staleness_ms"),
            product_isolation_passed=_require_bool(product_isolation_passed, "product_isolation_passed"),
            private_payload_exposed=_require_bool(private_payload_exposed, "private_payload_exposed"),
            explanation_trace_sha256=require_sha256(
                explanation_trace_sha256, "explanation_trace_sha256"
            ),
            observed_at=normalize_timestamp(observed_at, "observed_at"),
            observation_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "observation_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope.metadata_record(),
            "workload_id": self.workload_id,
            "candidate_id": self.candidate_id,
            "result_record_ids": list(self.result_record_ids),
            "conflict_record_ids": list(self.conflict_record_ids),
            "correction_record_ids": list(self.correction_record_ids),
            "deleted_record_ids_returned": list(self.deleted_record_ids_returned),
            "latency_ms": self.latency_ms,
            "staleness_ms": self.staleness_ms,
            "product_isolation_passed": self.product_isolation_passed,
            "private_payload_exposed": self.private_payload_exposed,
            "explanation_trace_sha256": self.explanation_trace_sha256,
            "observed_at": self.observed_at,
        }
        if include_digest:
            payload["observation_sha256"] = self.observation_sha256
        return payload

    def validate(self) -> None:
        recreated = ShadowReadObservation.create(
            scope=self.scope,
            workload_id=self.workload_id,
            candidate_id=self.candidate_id,
            result_record_ids=self.result_record_ids,
            conflict_record_ids=self.conflict_record_ids,
            correction_record_ids=self.correction_record_ids,
            deleted_record_ids_returned=self.deleted_record_ids_returned,
            latency_ms=self.latency_ms,
            staleness_ms=self.staleness_ms,
            product_isolation_passed=self.product_isolation_passed,
            private_payload_exposed=self.private_payload_exposed,
            explanation_trace_sha256=self.explanation_trace_sha256,
            observed_at=self.observed_at,
        )
        if recreated.observation_sha256 != require_sha256(
            self.observation_sha256, "observation_sha256"
        ):
            raise CognitiveKernelContractError("shadow-read observation digest mismatch")


@dataclass(frozen=True)
class ShadowReadComparisonReceipt:
    workload_id: str
    baseline_candidate_id: str
    candidate_id: str
    result_equivalent: bool
    authority_correct: bool
    conflict_correct: bool
    correction_correct: bool
    deletion_correct: bool
    privacy_correct: bool
    product_isolation_correct: bool
    latency_delta_ms: float
    staleness_delta_ms: float
    state: str
    compared_at: str
    production_influence: bool
    comparison_sha256: str

    @classmethod
    def create(
        cls,
        *,
        workload: ShadowReadWorkload,
        baseline: ShadowReadObservation,
        candidate: ShadowReadObservation,
        compared_at: object,
    ) -> "ShadowReadComparisonReceipt":
        workload.validate()
        baseline.validate()
        candidate.validate()
        if baseline.scope != workload.scope or candidate.scope != workload.scope:
            raise CognitiveKernelContractError("shadow-read scope mismatch")
        if baseline.workload_id != workload.workload_id or candidate.workload_id != workload.workload_id:
            raise CognitiveKernelContractError("shadow-read workload identity mismatch")
        if baseline.candidate_id == candidate.candidate_id:
            raise CognitiveKernelContractError("baseline and candidate identities must differ")
        expected = set(workload.expected_record_ids)
        result_equivalent = set(candidate.result_record_ids) == expected
        authority_correct = candidate.scope == workload.scope
        conflict_correct = set(candidate.conflict_record_ids) == set(
            workload.expected_conflict_record_ids
        )
        correction_correct = set(candidate.correction_record_ids) == set(
            workload.expected_correction_record_ids
        )
        deletion_correct = not (
            set(candidate.result_record_ids) & set(workload.expected_deleted_record_ids)
        ) and not candidate.deleted_record_ids_returned
        privacy_correct = not candidate.private_payload_exposed
        isolation_correct = candidate.product_isolation_passed
        correctness = all(
            (
                result_equivalent,
                authority_correct,
                conflict_correct,
                correction_correct,
                deletion_correct,
                privacy_correct,
                isolation_correct,
            )
        )
        baseline_correct = (
            set(baseline.result_record_ids) == expected
            and not baseline.private_payload_exposed
            and baseline.product_isolation_passed
        )
        latency_delta = candidate.latency_ms - baseline.latency_ms
        staleness_delta = candidate.staleness_ms - baseline.staleness_ms
        if not correctness:
            state = "candidate_degraded"
        elif not baseline_correct or latency_delta < 0 or staleness_delta < 0:
            state = "candidate_improved"
        elif latency_delta == 0 and staleness_delta == 0:
            state = "equivalent"
        else:
            state = "inconclusive"
        record = cls(
            workload_id=workload.workload_id,
            baseline_candidate_id=baseline.candidate_id,
            candidate_id=candidate.candidate_id,
            result_equivalent=result_equivalent,
            authority_correct=authority_correct,
            conflict_correct=conflict_correct,
            correction_correct=correction_correct,
            deletion_correct=deletion_correct,
            privacy_correct=privacy_correct,
            product_isolation_correct=isolation_correct,
            latency_delta_ms=latency_delta,
            staleness_delta_ms=staleness_delta,
            state=_require_choice(state, "state", SHADOW_COMPARISON_STATES),
            compared_at=normalize_timestamp(compared_at, "compared_at"),
            production_influence=False,
            comparison_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "comparison_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "workload_id": self.workload_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "candidate_id": self.candidate_id,
            "result_equivalent": self.result_equivalent,
            "authority_correct": self.authority_correct,
            "conflict_correct": self.conflict_correct,
            "correction_correct": self.correction_correct,
            "deletion_correct": self.deletion_correct,
            "privacy_correct": self.privacy_correct,
            "product_isolation_correct": self.product_isolation_correct,
            "latency_delta_ms": self.latency_delta_ms,
            "staleness_delta_ms": self.staleness_delta_ms,
            "state": self.state,
            "compared_at": self.compared_at,
            "production_influence": self.production_influence,
        }
        if include_digest:
            payload["comparison_sha256"] = self.comparison_sha256
        return payload


@dataclass(frozen=True)
class DestinationCandidateEvaluation:
    candidate_id: str
    comparison_receipt_ids: tuple[str, ...]
    workload_count: int
    equivalent_count: int
    improved_count: int
    degraded_count: int
    inconclusive_count: int
    recommendation: str
    evaluated_at: str
    production_selection: bool
    evaluation_sha256: str

    @classmethod
    def create(
        cls,
        *,
        profile: DestinationCandidateProfile,
        comparisons: tuple[ShadowReadComparisonReceipt, ...] | list[ShadowReadComparisonReceipt],
        evaluated_at: object,
    ) -> "DestinationCandidateEvaluation":
        profile.validate()
        items = tuple(comparisons)
        if not items:
            raise CognitiveKernelContractError("at least one comparison is required")
        for item in items:
            if item.candidate_id != profile.candidate_id:
                raise CognitiveKernelContractError("comparison candidate mismatch")
        counts = {state: 0 for state in SHADOW_COMPARISON_STATES}
        for item in items:
            counts[item.state] += 1
        if counts["candidate_degraded"]:
            recommendation = "repair_before_continuation"
        elif counts["inconclusive"]:
            recommendation = "continue_shadow_evaluation"
        else:
            recommendation = "eligible_for_next_research_gate"
        record = cls(
            candidate_id=profile.candidate_id,
            comparison_receipt_ids=tuple(item.comparison_sha256 for item in items),
            workload_count=len(items),
            equivalent_count=counts["equivalent"],
            improved_count=counts["candidate_improved"],
            degraded_count=counts["candidate_degraded"],
            inconclusive_count=counts["inconclusive"],
            recommendation=_require_choice(
                recommendation, "recommendation", DESTINATION_RECOMMENDATIONS
            ),
            evaluated_at=normalize_timestamp(evaluated_at, "evaluated_at"),
            production_selection=False,
            evaluation_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "evaluation_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "comparison_receipt_ids": list(self.comparison_receipt_ids),
            "workload_count": self.workload_count,
            "equivalent_count": self.equivalent_count,
            "improved_count": self.improved_count,
            "degraded_count": self.degraded_count,
            "inconclusive_count": self.inconclusive_count,
            "recommendation": self.recommendation,
            "evaluated_at": self.evaluated_at,
            "production_selection": self.production_selection,
        }
        if include_digest:
            payload["evaluation_sha256"] = self.evaluation_sha256
        return payload


def build_m2_destination_candidate_profile(
    *, scope: ProductHostScope, created_at: object
) -> DestinationCandidateProfile:
    """Describe the existing M2 reversible prototypes as one read-only candidate."""
    return DestinationCandidateProfile.create(
        scope=scope,
        candidate_id="m2.reversible.polyglot.candidate",
        backend_types=(
            "embedded_sqlite",
            "content_addressed_object",
            "graph_projection",
            "vector_projection",
            "durable_workflow",
        ),
        component_ids=(
            "claim_authority.prototype",
            "shadow_adjudication.prototype",
            "projection.prototype",
            "bounded_serving.prototype",
            "durable_workflow.prototype",
            "deletion_propagation.prototype",
        ),
        contract_roles=(
            "evidence_log",
            "claim_authority",
            "current_claim_projection",
            "graph_projection",
            "vector_projection",
            "payload_store",
            "deletion_coordinator",
        ),
        deployment_profiles=(
            "embedded_edge",
            "single_workstation",
            "distributed_multi_region",
            "frontier_research",
        ),
        created_at=created_at,
    )
