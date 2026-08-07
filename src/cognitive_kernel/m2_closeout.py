"""M2-CLOSEOUT integration-evaluation and shadow-migration admission contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_sha256,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope

M2_CLOSEOUT_SCHEMA_VERSION = "1.0.0"

M2_CLOSEOUT_COMPONENTS = frozenset(
    {
        "foundation_contracts",
        "claim_authority",
        "shadow_adjudication",
        "projection_fabric",
        "bounded_serving",
        "durable_workflow",
        "deletion_propagation",
        "cross_prototype_lineage",
    }
)
M2_COMPONENT_EVALUATION_STATES = frozenset(
    {"passed", "failed", "blocked", "not_evaluated"}
)
SHADOW_MIGRATION_ADMISSION_OUTCOMES = frozenset(
    {
        "denied",
        "conditional",
        "admitted_preparatory_read_only",
    }
)
SHADOW_MIGRATION_STAGES = frozenset(
    {
        "stage_a_inventory_registration",
        "stage_b_contract_adapters",
        "stage_c_destination_candidates",
        "stage_d_historical_backfill",
        "stage_e_shadow_reads",
        "stage_f_controlled_write_mirroring",
        "stage_g_graph_vector_workflow_build",
        "stage_h_canary_authority",
        "stage_i_cutover",
        "stage_j_compatibility_operation",
    }
)

_PREPARATORY_STAGES = frozenset(
    {
        "stage_a_inventory_registration",
        "stage_b_contract_adapters",
        "stage_c_destination_candidates",
        "stage_e_shadow_reads",
    }
)
_EXCLUDED_ACTIVATION_STAGES = frozenset(
    {
        "stage_d_historical_backfill",
        "stage_f_controlled_write_mirroring",
        "stage_g_graph_vector_workflow_build",
        "stage_h_canary_authority",
        "stage_i_cutover",
        "stage_j_compatibility_operation",
    }
)


def _sorted_identifiers(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    return tuple(sorted(normalize_identifier_sequence(values, field)))


def _require_envelope(
    envelope: MemoryUnitEnvelope,
    *,
    record_type: str,
    record_id: str,
) -> None:
    if not isinstance(envelope, MemoryUnitEnvelope):
        raise CognitiveKernelContractError(
            "envelope must be MemoryUnitEnvelope"
        )
    envelope.validate()
    if envelope.record_type != record_type:
        raise CognitiveKernelContractError(
            f"envelope.record_type must be {record_type}"
        )
    if envelope.record_id != record_id:
        raise CognitiveKernelContractError(
            "envelope.record_id must equal the contract record id"
        )
    if envelope.authority_role != "evaluation_artifact":
        raise CognitiveKernelContractError(
            "M2 closeout records require evaluation_artifact authority"
        )


@dataclass(frozen=True)
class M2ComponentEvaluation:
    """One component result in the M2-CLOSEOUT integration evaluation."""

    component_id: str
    state: str
    evidence_record_ids: tuple[str, ...]
    evaluated_at: str
    details_content_digest: str
    result_sha256: str

    @classmethod
    def create(
        cls,
        *,
        component_id: object,
        state: object,
        evidence_record_ids: Iterable[object],
        evaluated_at: object,
        details_content: Mapping[str, object],
    ) -> "M2ComponentEvaluation":
        normalized_component = require_identifier(
            component_id, "component_id"
        )
        normalized_state = require_identifier(state, "state")
        if normalized_component not in M2_CLOSEOUT_COMPONENTS:
            raise CognitiveKernelContractError(
                "component_id is not an M2 closeout component"
            )
        if normalized_state not in M2_COMPONENT_EVALUATION_STATES:
            raise CognitiveKernelContractError(
                "state is not an M2 component evaluation state"
            )
        if not isinstance(details_content, Mapping):
            raise CognitiveKernelContractError(
                "details_content must be a mapping"
            )
        material = {
            "component_id": normalized_component,
            "state": normalized_state,
            "evidence_record_ids": list(
                _sorted_identifiers(
                    evidence_record_ids,
                    "evidence_record_ids",
                )
            ),
            "evaluated_at": normalize_timestamp(
                evaluated_at, "evaluated_at"
            ),
            "details_content_digest": canonical_sha256(
                dict(details_content)
            ),
        }
        return cls(
            component_id=material["component_id"],
            state=material["state"],
            evidence_record_ids=tuple(
                material["evidence_record_ids"]
            ),
            evaluated_at=material["evaluated_at"],
            details_content_digest=material[
                "details_content_digest"
            ],
            result_sha256=canonical_sha256(material),
        )

    def material_record(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "state": self.state,
            "evidence_record_ids": list(self.evidence_record_ids),
            "evaluated_at": self.evaluated_at,
            "details_content_digest": self.details_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.material_record(),
            "result_sha256": self.result_sha256,
        }

    def validate(self) -> None:
        rebuilt = self.create(
            component_id=self.component_id,
            state=self.state,
            evidence_record_ids=self.evidence_record_ids,
            evaluated_at=self.evaluated_at,
            details_content={
                "digest_reference": self.details_content_digest
            },
        )
        # The details are intentionally external to this contract. Validate
        # the stable fields and the stored digest directly.
        require_sha256(
            self.details_content_digest,
            "details_content_digest",
        )
        expected = canonical_sha256(self.material_record())
        if expected != self.result_sha256:
            raise CognitiveKernelContractError(
                "M2 component evaluation digest mismatch"
            )
        if rebuilt.component_id != self.component_id:
            raise CognitiveKernelContractError(
                "M2 component evaluation is not canonical"
            )


@dataclass(frozen=True)
class M2CloseoutReport:
    """Synthetic cross-prototype M2 closeout evidence."""

    envelope: MemoryUnitEnvelope
    evaluation_id: str
    component_results: tuple[M2ComponentEvaluation, ...]
    cross_prototype_record_ids: tuple[str, ...]
    evaluated_at: str
    production_influence: bool
    canonical_authority_transfer: bool
    private_payload_read: bool
    phase2_migration_started: bool
    report_content_digest: str
    report_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        evaluation_id: object,
        component_results: Iterable[M2ComponentEvaluation],
        cross_prototype_record_ids: Iterable[object],
        evaluated_at: object,
        full_report_content: Mapping[str, object],
        production_influence: object = False,
        canonical_authority_transfer: object = False,
        private_payload_read: object = False,
        phase2_migration_started: object = False,
    ) -> "M2CloseoutReport":
        normalized_id = require_identifier(
            evaluation_id, "evaluation_id"
        )
        _require_envelope(
            envelope,
            record_type="m2_closeout_report",
            record_id=normalized_id,
        )
        results = tuple(component_results)
        if not results:
            raise CognitiveKernelContractError(
                "component_results may not be empty"
            )
        by_component: dict[str, M2ComponentEvaluation] = {}
        for result in results:
            if not isinstance(result, M2ComponentEvaluation):
                raise CognitiveKernelContractError(
                    "component_results must contain M2ComponentEvaluation"
                )
            result.validate()
            if result.component_id in by_component:
                raise CognitiveKernelContractError(
                    "component_results may not duplicate components"
                )
            by_component[result.component_id] = result
        missing = sorted(M2_CLOSEOUT_COMPONENTS - set(by_component))
        if missing:
            raise CognitiveKernelContractError(
                f"component_results are incomplete: {missing}"
            )
        if any(item.state != "passed" for item in results):
            raise CognitiveKernelContractError(
                "M2 closeout requires every component to pass"
            )
        for field, value in (
            ("production_influence", production_influence),
            ("canonical_authority_transfer", canonical_authority_transfer),
            ("private_payload_read", private_payload_read),
            ("phase2_migration_started", phase2_migration_started),
        ):
            if value is not False:
                raise CognitiveKernelContractError(
                    f"{field} violates the M2 closeout evaluation safety profile"
                )
        if not isinstance(full_report_content, Mapping):
            raise CognitiveKernelContractError(
                "full_report_content must be a mapping"
            )
        material = {
            "envelope": envelope.metadata_record(),
            "evaluation_id": normalized_id,
            "component_results": [
                item.metadata_record()
                for item in sorted(
                    results,
                    key=lambda item: item.component_id,
                )
            ],
            "cross_prototype_record_ids": list(
                _sorted_identifiers(
                    cross_prototype_record_ids,
                    "cross_prototype_record_ids",
                )
            ),
            "evaluated_at": normalize_timestamp(
                evaluated_at, "evaluated_at"
            ),
            "production_influence": False,
            "canonical_authority_transfer": False,
            "private_payload_read": False,
            "phase2_migration_started": False,
            "report_content_digest": canonical_sha256(
                dict(full_report_content)
            ),
        }
        return cls(
            envelope=envelope,
            evaluation_id=normalized_id,
            component_results=tuple(
                sorted(results, key=lambda item: item.component_id)
            ),
            cross_prototype_record_ids=tuple(
                material["cross_prototype_record_ids"]
            ),
            evaluated_at=material["evaluated_at"],
            production_influence=False,
            canonical_authority_transfer=False,
            private_payload_read=False,
            phase2_migration_started=False,
            report_content_digest=material[
                "report_content_digest"
            ],
            report_sha256=canonical_sha256(material),
        )

    def material_record(self) -> dict[str, object]:
        return {
            "envelope": self.envelope.metadata_record(),
            "evaluation_id": self.evaluation_id,
            "component_results": [
                item.metadata_record()
                for item in self.component_results
            ],
            "cross_prototype_record_ids": list(
                self.cross_prototype_record_ids
            ),
            "evaluated_at": self.evaluated_at,
            "production_influence": self.production_influence,
            "canonical_authority_transfer": (
                self.canonical_authority_transfer
            ),
            "private_payload_read": self.private_payload_read,
            "phase2_migration_started": (
                self.phase2_migration_started
            ),
            "report_content_digest": self.report_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.material_record(),
            "report_sha256": self.report_sha256,
        }

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_type="m2_closeout_report",
            record_id=self.evaluation_id,
        )
        if set(
            item.component_id for item in self.component_results
        ) != M2_CLOSEOUT_COMPONENTS:
            raise CognitiveKernelContractError(
                "M2 closeout component set mismatch"
            )
        for item in self.component_results:
            item.validate()
            if item.state != "passed":
                raise CognitiveKernelContractError(
                    "M2 closeout contains a non-passing component"
                )
        if any(
            (
                self.production_influence,
                self.canonical_authority_transfer,
                self.private_payload_read,
                self.phase2_migration_started,
            )
        ):
            raise CognitiveKernelContractError(
                "M2 closeout evaluation crossed an activation boundary"
            )
        require_sha256(
            self.report_content_digest,
            "report_content_digest",
        )
        if canonical_sha256(self.material_record()) != self.report_sha256:
            raise CognitiveKernelContractError(
                "M2 closeout report digest mismatch"
            )


@dataclass(frozen=True)
class ShadowMigrationAdmissionDecision:
    """Admission review for preparatory read-only Phase 2 shadow work."""

    envelope: MemoryUnitEnvelope
    decision_id: str
    evaluation_id: str
    outcome: str
    admitted_stages: tuple[str, ...]
    excluded_stages: tuple[str, ...]
    reason_codes: tuple[str, ...]
    decided_at: str
    phase2_migration_started: bool
    p5_1e_unblocked: bool
    production_write_mirroring: bool
    canonical_authority_transfer: bool
    private_payload_read: bool
    decision_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        decision_id: object,
        evaluation_id: object,
        outcome: object,
        admitted_stages: Iterable[object],
        excluded_stages: Iterable[object],
        reason_codes: Iterable[object],
        decided_at: object,
        phase2_migration_started: object = False,
        p5_1e_unblocked: object = False,
        production_write_mirroring: object = False,
        canonical_authority_transfer: object = False,
        private_payload_read: object = False,
    ) -> "ShadowMigrationAdmissionDecision":
        normalized_id = require_identifier(
            decision_id, "decision_id"
        )
        normalized_evaluation = require_identifier(
            evaluation_id, "evaluation_id"
        )
        normalized_outcome = require_identifier(
            outcome, "outcome"
        )
        if normalized_outcome not in (
            SHADOW_MIGRATION_ADMISSION_OUTCOMES
        ):
            raise CognitiveKernelContractError(
                "outcome is not a shadow-migration admission outcome"
            )
        _require_envelope(
            envelope,
            record_type="shadow_migration_admission",
            record_id=normalized_id,
        )
        admitted = _sorted_identifiers(
            admitted_stages, "admitted_stages"
        )
        excluded = _sorted_identifiers(
            excluded_stages, "excluded_stages"
        )
        unknown = (
            set(admitted) | set(excluded)
        ) - SHADOW_MIGRATION_STAGES
        if unknown:
            raise CognitiveKernelContractError(
                f"unknown shadow migration stages: {sorted(unknown)}"
            )
        if set(admitted) & set(excluded):
            raise CognitiveKernelContractError(
                "admitted and excluded stages must not overlap"
            )
        if normalized_outcome == "admitted_preparatory_read_only":
            if set(admitted) != _PREPARATORY_STAGES:
                raise CognitiveKernelContractError(
                    "preparatory admission stage set is incomplete"
                )
            if set(excluded) != _EXCLUDED_ACTIVATION_STAGES:
                raise CognitiveKernelContractError(
                    "preparatory exclusion stage set is incomplete"
                )
        for field, value in (
            ("phase2_migration_started", phase2_migration_started),
            ("p5_1e_unblocked", p5_1e_unblocked),
            ("production_write_mirroring", production_write_mirroring),
            ("canonical_authority_transfer", canonical_authority_transfer),
            ("private_payload_read", private_payload_read),
        ):
            if value is not False:
                raise CognitiveKernelContractError(
                    f"{field} violates the admission-review safety profile"
                )
        material = {
            "envelope": envelope.metadata_record(),
            "decision_id": normalized_id,
            "evaluation_id": normalized_evaluation,
            "outcome": normalized_outcome,
            "admitted_stages": list(admitted),
            "excluded_stages": list(excluded),
            "reason_codes": list(
                _sorted_identifiers(reason_codes, "reason_codes")
            ),
            "decided_at": normalize_timestamp(
                decided_at, "decided_at"
            ),
            "phase2_migration_started": False,
            "p5_1e_unblocked": False,
            "production_write_mirroring": False,
            "canonical_authority_transfer": False,
            "private_payload_read": False,
        }
        return cls(
            envelope=envelope,
            decision_id=normalized_id,
            evaluation_id=normalized_evaluation,
            outcome=normalized_outcome,
            admitted_stages=admitted,
            excluded_stages=excluded,
            reason_codes=tuple(material["reason_codes"]),
            decided_at=material["decided_at"],
            phase2_migration_started=False,
            p5_1e_unblocked=False,
            production_write_mirroring=False,
            canonical_authority_transfer=False,
            private_payload_read=False,
            decision_sha256=canonical_sha256(material),
        )

    def material_record(self) -> dict[str, object]:
        return {
            "envelope": self.envelope.metadata_record(),
            "decision_id": self.decision_id,
            "evaluation_id": self.evaluation_id,
            "outcome": self.outcome,
            "admitted_stages": list(self.admitted_stages),
            "excluded_stages": list(self.excluded_stages),
            "reason_codes": list(self.reason_codes),
            "decided_at": self.decided_at,
            "phase2_migration_started": (
                self.phase2_migration_started
            ),
            "p5_1e_unblocked": self.p5_1e_unblocked,
            "production_write_mirroring": (
                self.production_write_mirroring
            ),
            "canonical_authority_transfer": (
                self.canonical_authority_transfer
            ),
            "private_payload_read": self.private_payload_read,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.material_record(),
            "decision_sha256": self.decision_sha256,
        }

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_type="shadow_migration_admission",
            record_id=self.decision_id,
        )
        if self.outcome not in SHADOW_MIGRATION_ADMISSION_OUTCOMES:
            raise CognitiveKernelContractError(
                "invalid shadow migration admission outcome"
            )
        if self.outcome == "admitted_preparatory_read_only":
            if set(self.admitted_stages) != _PREPARATORY_STAGES:
                raise CognitiveKernelContractError(
                    "preparatory admission stage set mismatch"
                )
            if set(self.excluded_stages) != (
                _EXCLUDED_ACTIVATION_STAGES
            ):
                raise CognitiveKernelContractError(
                    "preparatory exclusion stage set mismatch"
                )
        if any(
            (
                self.phase2_migration_started,
                self.p5_1e_unblocked,
                self.production_write_mirroring,
                self.canonical_authority_transfer,
                self.private_payload_read,
            )
        ):
            raise CognitiveKernelContractError(
                "admission decision crossed an activation boundary"
            )
        if canonical_sha256(self.material_record()) != self.decision_sha256:
            raise CognitiveKernelContractError(
                "shadow migration admission digest mismatch"
            )

    def assert_supported_by(
        self,
        report: M2CloseoutReport,
    ) -> None:
        if not isinstance(report, M2CloseoutReport):
            raise CognitiveKernelContractError(
                "report must be M2CloseoutReport"
            )
        report.validate()
        self.validate()
        if self.evaluation_id != report.evaluation_id:
            raise CognitiveKernelContractError(
                "admission decision references another evaluation"
            )
        if self.outcome == "admitted_preparatory_read_only":
            if any(
                (
                    report.production_influence,
                    report.canonical_authority_transfer,
                    report.private_payload_read,
                    report.phase2_migration_started,
                )
            ):
                raise CognitiveKernelContractError(
                    "admission is not supported by a contained report"
                )
