"""Loader for the P5.1c retention lifecycle decision-journal policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    require_identifier,
    require_schema_version,
    require_text,
)
from .lifecycle import (
    ALLOWED_LIFECYCLE_TRANSITIONS,
    LIFECYCLE_AUTHORITY_LEVELS,
    LIFECYCLE_AUTHORITY_REQUIREMENTS,
    LIFECYCLE_DECISION_OUTCOMES,
    LIFECYCLE_DECISION_TYPES,
    RETENTION_BLOCKER_STATES,
    RETENTION_BLOCKER_TYPES,
)

_POLICY_KEYS = {
    "cognitive_kernel_lifecycle_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "required_contracts",
    "database_format",
    "journal_schema_version",
    "journal_mode",
    "synchronous",
    "hash_algorithm",
    "required_scope_fields",
    "decision_types",
    "decision_outcomes",
    "blocker_types",
    "blocker_states",
    "authority_levels",
    "transition_matrix",
    "authority_requirements",
    "invariants",
    "capability_ceiling",
}
_INVARIANT_KEYS = {
    "metadata_only_records",
    "logically_append_only",
    "deterministic_record_identity",
    "deterministic_sequence_numbers",
    "tamper_evident_hash_chain",
    "duplicate_record_rejection",
    "authorized_transition_validation",
    "retention_blocker_lineage",
    "authorized_override_lineage",
    "database_outside_public_repository",
    "product_host_encryption_isolation",
    "atomic_transactions",
    "full_integrity_verification",
    "sanitized_inspection",
    "automatic_expiry_implemented",
    "physical_tier_movement_implemented",
    "payload_deletion_implemented",
    "deletion_propagation_implemented",
    "storage_pressure_eviction_implemented",
    "backup_restore_implemented",
    "learning_curator_implemented",
    "phase2_memory_store_migrated",
    "friday_product_source_implemented",
    "private_payload_allowed",
    "kernel_key_custody",
}
_REQUIRED_CONTRACTS = ("experience_event", "storage_lifecycle")
_REQUIRED_SCOPE_FIELDS = (
    "product_id",
    "host_instance_id",
    "encryption_domain",
)
_REQUIRED_TRUE = {
    "metadata_only_records",
    "logically_append_only",
    "deterministic_record_identity",
    "deterministic_sequence_numbers",
    "tamper_evident_hash_chain",
    "duplicate_record_rejection",
    "authorized_transition_validation",
    "retention_blocker_lineage",
    "authorized_override_lineage",
    "database_outside_public_repository",
    "product_host_encryption_isolation",
    "atomic_transactions",
    "full_integrity_verification",
    "sanitized_inspection",
}


@dataclass(frozen=True)
class CognitiveKernelLifecyclePolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_contracts: tuple[str, ...]
    database_format: str
    journal_schema_version: str
    journal_mode: str
    synchronous: str
    hash_algorithm: str
    required_scope_fields: tuple[str, ...]
    decision_types: tuple[str, ...]
    decision_outcomes: tuple[str, ...]
    blocker_types: tuple[str, ...]
    blocker_states: tuple[str, ...]
    authority_levels: tuple[str, ...]
    transition_matrix: Mapping[str, tuple[str, ...]]
    authority_requirements: Mapping[str, str]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "lifecycle policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        if require_schema_version(self.version, "version") != "0.8.0":
            raise CognitiveKernelContractError(
                "lifecycle policy version changed"
            )
        if self.phase != "5" or self.milestone != "P5.1c":
            raise CognitiveKernelContractError(
                "lifecycle policy milestone changed"
            )
        if self.status != "runtime_foundation":
            raise CognitiveKernelContractError(
                "lifecycle policy status is invalid"
            )
        if self.required_contracts != _REQUIRED_CONTRACTS:
            raise CognitiveKernelContractError(
                "lifecycle required contracts changed"
            )
        if self.database_format != "sqlite3":
            raise CognitiveKernelContractError(
                "lifecycle database format changed"
            )
        if self.journal_schema_version != "1.0.0":
            raise CognitiveKernelContractError(
                "lifecycle journal schema version changed"
            )
        if self.journal_mode != "wal" or self.synchronous != "full":
            raise CognitiveKernelContractError(
                "lifecycle durability settings changed"
            )
        if self.hash_algorithm != "sha256":
            raise CognitiveKernelContractError(
                "lifecycle hash algorithm changed"
            )
        if self.required_scope_fields != _REQUIRED_SCOPE_FIELDS:
            raise CognitiveKernelContractError(
                "lifecycle scope fields changed"
            )
        if set(self.decision_types) != set(LIFECYCLE_DECISION_TYPES):
            raise CognitiveKernelContractError(
                "lifecycle decision types changed"
            )
        if set(self.decision_outcomes) != set(LIFECYCLE_DECISION_OUTCOMES):
            raise CognitiveKernelContractError(
                "lifecycle decision outcomes changed"
            )
        if set(self.blocker_types) != set(RETENTION_BLOCKER_TYPES):
            raise CognitiveKernelContractError(
                "retention blocker types changed"
            )
        if set(self.blocker_states) != set(RETENTION_BLOCKER_STATES):
            raise CognitiveKernelContractError(
                "retention blocker states changed"
            )
        if self.authority_levels != tuple(LIFECYCLE_AUTHORITY_LEVELS):
            raise CognitiveKernelContractError(
                "lifecycle authority levels changed"
            )
        expected_matrix = {
            source: tuple(sorted(targets))
            for source, targets in ALLOWED_LIFECYCLE_TRANSITIONS.items()
        }
        if dict(self.transition_matrix) != expected_matrix:
            raise CognitiveKernelContractError(
                "lifecycle transition matrix changed"
            )
        if dict(self.authority_requirements) != dict(
            LIFECYCLE_AUTHORITY_REQUIREMENTS
        ):
            raise CognitiveKernelContractError(
                "lifecycle authority requirements changed"
            )
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError(
                "lifecycle invariants changed"
            )
        for key in _REQUIRED_TRUE:
            if self.invariants.get(key) is not True:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be true"
                )
        for key in _INVARIANT_KEYS - _REQUIRED_TRUE:
            if self.invariants.get(key) is not False:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be false"
                )
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError(
                "lifecycle policy may not be a capability ceiling"
            )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CognitiveKernelContractError(
            f"could not load JSON policy: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(
            f"JSON policy must be an object: {path}"
        )
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise CognitiveKernelContractError(
            f"{field} keys changed; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _sequence(
    value: object,
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CognitiveKernelContractError(
            f"{field} must be a list"
        )
    result = tuple(require_text(item, field, maximum=128) for item in value)
    if len(set(result)) != len(result):
        raise CognitiveKernelContractError(
            f"{field} may not contain duplicates"
        )
    return result


def _require_dict(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(f"{field} must be an object")
    return value


def load_cognitive_kernel_lifecycle_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelLifecyclePolicy:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = (
        Path(path).resolve()
        if path is not None
        else root / "policies" / "cognitive_kernel_lifecycle_policy.json"
    )
    payload = _read_json(source)
    _exact_keys(payload, _POLICY_KEYS, "lifecycle policy")
    invariants = _require_dict(payload["invariants"], "invariants")
    _exact_keys(invariants, _INVARIANT_KEYS, "lifecycle invariants")
    if any(not isinstance(value, bool) for value in invariants.values()):
        raise CognitiveKernelContractError(
            "lifecycle invariants must be booleans"
        )
    matrix_payload = _require_dict(
        payload["transition_matrix"], "transition_matrix"
    )
    if set(matrix_payload) != set(ALLOWED_LIFECYCLE_TRANSITIONS):
        raise CognitiveKernelContractError(
            "transition-matrix source tiers changed"
        )
    transition_matrix = {
        require_identifier(source_tier, "source_tier"): tuple(
            sorted(
                _sequence(
                    targets,
                    f"transition_matrix.{source_tier}",
                    allow_empty=True,
                )
            )
        )
        for source_tier, targets in matrix_payload.items()
    }
    authority_payload = _require_dict(
        payload["authority_requirements"], "authority_requirements"
    )
    authority_requirements = {
        require_identifier(key, "authority_requirement"): require_identifier(
            value, "authority_level"
        )
        for key, value in authority_payload.items()
    }
    policy = CognitiveKernelLifecyclePolicy(
        schema_version=int(
            payload["cognitive_kernel_lifecycle_policy_schema_version"]
        ),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(
            payload["milestone"], "milestone", maximum=32
        ),
        status=require_identifier(payload["status"], "status"),
        required_contracts=_sequence(
            payload["required_contracts"], "required_contracts"
        ),
        database_format=require_identifier(
            payload["database_format"], "database_format"
        ),
        journal_schema_version=require_schema_version(
            payload["journal_schema_version"], "journal_schema_version"
        ),
        journal_mode=require_identifier(
            payload["journal_mode"], "journal_mode"
        ),
        synchronous=require_identifier(
            payload["synchronous"], "synchronous"
        ),
        hash_algorithm=require_identifier(
            payload["hash_algorithm"], "hash_algorithm"
        ),
        required_scope_fields=_sequence(
            payload["required_scope_fields"], "required_scope_fields"
        ),
        decision_types=_sequence(
            payload["decision_types"], "decision_types"
        ),
        decision_outcomes=_sequence(
            payload["decision_outcomes"], "decision_outcomes"
        ),
        blocker_types=_sequence(
            payload["blocker_types"], "blocker_types"
        ),
        blocker_states=_sequence(
            payload["blocker_states"], "blocker_states"
        ),
        authority_levels=_sequence(
            payload["authority_levels"], "authority_levels"
        ),
        transition_matrix=transition_matrix,
        authority_requirements=authority_requirements,
        invariants={str(key): bool(value) for key, value in invariants.items()},
        capability_ceiling=bool(payload["capability_ceiling"]),
        digest=canonical_sha256(payload),
        source_path=source,
    )
    policy.validate()

    storage = _read_json(root / "policies" / "storage_lifecycle_policy.json")
    if storage.get("permanent_compact_event_ledger") is not True:
        raise CognitiveKernelContractError(
            "storage policy does not require a permanent compact ledger"
        )
    tiers = _require_dict(storage.get("tiers"), "tiers")
    if set(tiers) != set(ALLOWED_LIFECYCLE_TRANSITIONS):
        raise CognitiveKernelContractError("storage tier vocabulary changed")
    retention_classes = _require_dict(
        storage.get("retention_classes"), "retention_classes"
    )
    expected_classes = {
        "authoritative_source",
        "active_project",
        "high_value_experience",
        "ordinary_experience",
        "transient_web_or_tool_cache",
        "failed_experiment",
        "training_replay",
        "quarantine",
        "owner_hold",
    }
    if set(retention_classes) != expected_classes:
        raise CognitiveKernelContractError(
            "storage retention-class vocabulary changed"
        )
    if tuple(storage.get("retention_blockers", ())) != tuple(
        sorted(RETENTION_BLOCKER_TYPES)
    ):
        source_blockers = storage.get("retention_blockers")
        if not isinstance(source_blockers, list) or set(source_blockers) != set(
            RETENTION_BLOCKER_TYPES
        ):
            raise CognitiveKernelContractError(
                "storage retention-blocker vocabulary changed"
            )
    capacity = _require_dict(storage.get("capacity"), "capacity")
    if capacity.get("protected_artifact_silent_deletion_allowed") is not False:
        raise CognitiveKernelContractError(
            "storage policy permits silent protected-artifact deletion"
        )
    deletion = _require_dict(storage.get("deletion"), "deletion")
    if deletion.get("deliberate_relearning_of_deleted_payload_allowed") is not False:
        raise CognitiveKernelContractError(
            "storage policy permits deliberate relearning of deleted payloads"
        )

    foundation = _read_json(
        root / "policies" / "cognitive_kernel_foundation_policy.json"
    )
    if foundation.get("version") != "0.1.0" or foundation.get(
        "milestone"
    ) != "P5.0b":
        raise CognitiveKernelContractError(
            "unexpected Cognitive Kernel foundation baseline"
        )
    boundaries = _require_dict(foundation.get("boundaries"), "boundaries")
    if boundaries.get("metadata_only_contracts") is not True:
        raise CognitiveKernelContractError(
            "Cognitive Kernel foundation is not metadata-only"
        )
    if boundaries.get("private_payload_allowed") is not False:
        raise CognitiveKernelContractError(
            "Cognitive Kernel foundation permits private payloads"
        )

    ledger_policy = _read_json(
        root / "policies" / "cognitive_kernel_experience_ledger_policy.json"
    )
    if ledger_policy.get("version") != "0.6.0" or ledger_policy.get(
        "milestone"
    ) != "P5.1a":
        raise CognitiveKernelContractError(
            "unexpected experience-ledger policy baseline"
        )
    raw_policy = _read_json(
        root / "policies" / "cognitive_kernel_raw_buffer_policy.json"
    )
    if raw_policy.get("version") != "0.7.0" or raw_policy.get(
        "milestone"
    ) != "P5.1b":
        raise CognitiveKernelContractError(
            "unexpected raw-buffer policy baseline"
        )

    profiles = _read_json(root / "policies" / "capability_profiles.json")
    if profiles.get("version") != "1.5.0":
        raise CognitiveKernelContractError(
            "capability profiles must be version 1.5.0"
        )
    profile_map = _require_dict(profiles.get("profiles"), "profiles")
    kernel = _require_dict(
        profile_map.get("kernel.phase5.foundation"),
        "kernel.phase5.foundation",
    )
    capabilities = _require_dict(
        kernel.get("capabilities"), "kernel capabilities"
    )
    for key in {
        "lifecycle_decision_contracts_implemented",
        "retention_blocker_contracts_implemented",
        "lifecycle_journal_runtime_implemented",
        "authorized_transition_validation_implemented",
        "append_only_override_lineage_implemented",
        "governed_non_destructive_tier_transition_runtime_implemented",
        "approved_transition_decision_binding_implemented",
        "blocker_revalidation_implemented",
        "source_preserving_copy_verify_publish_implemented",
        "tier_transition_crash_recovery_implemented",
        "tier_movement_implemented",
    }:
        if capabilities.get(key) is not True:
            raise CognitiveKernelContractError(
                f"kernel capability {key} must be true"
            )
    for key in {
        "automatic_retention_implemented",
        "payload_deletion_implemented",
        "deletion_propagation_implemented",
        "storage_pressure_eviction_implemented",
        "backup_restore_implemented",
        "learning_curator_implemented",
    }:
        if capabilities.get(key) is not False:
            raise CognitiveKernelContractError(
                f"kernel deferred capability {key} must be false"
            )

    product_lines = _read_json(root / "policies" / "product_lines.json")
    shared_kernel = _require_dict(
        product_lines.get("shared_kernel"), "shared_kernel"
    )
    contracts = shared_kernel.get("phase5_contracts")
    if not isinstance(contracts, list):
        raise CognitiveKernelContractError(
            "Phase 5 contract registry is invalid"
        )
    for contract in _REQUIRED_CONTRACTS:
        if contract not in contracts:
            raise CognitiveKernelContractError(
                f"Phase 5 contract registry is missing {contract}"
            )
    if shared_kernel.get("may_contain_personal_data") is not False:
        raise CognitiveKernelContractError(
            "shared kernel may contain personal data"
        )
    separation = _require_dict(
        product_lines.get("separation_rules"), "separation_rules"
    )
    if separation.get("cross_host_deduplication_allowed") is not False:
        raise CognitiveKernelContractError(
            "product policy permits cross-host deduplication"
        )
    if separation.get(
        "friday_product_source_may_live_in_alice_repository"
    ) is not False:
        raise CognitiveKernelContractError(
            "Friday product source is permitted in A.L.I.C.E."
        )
    return policy
