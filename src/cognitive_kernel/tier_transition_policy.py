"""Loader for the P5.1d governed tier-transition execution policy."""

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
)
from .tier_transition import (
    EXECUTABLE_SOURCE_TIERS,
    EXECUTABLE_TARGET_TIERS,
    TIER_TRANSITION_DECISION_TYPES,
    TIER_TRANSITION_SCHEMA_VERSION,
)

_POLICY_KEYS = {
    "cognitive_kernel_tier_transition_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "required_contracts",
    "store_schema_version",
    "database_format",
    "journal_mode",
    "synchronous",
    "hash_algorithm",
    "required_scope_fields",
    "executable_decision_types",
    "executable_source_tiers",
    "executable_target_tiers",
    "publication_protocol",
    "invariants",
    "capability_ceiling",
}

_INVARIANT_KEYS = {
    "approved_decision_binding",
    "decision_digest_binding",
    "lifecycle_journal_integrity_required",
    "retention_blocker_revalidation",
    "superseded_decision_rejection",
    "non_destructive_source_preservation",
    "host_sealed_opaque_bytes_only",
    "atomic_copy_verify_publish",
    "append_only_transition_intents",
    "append_only_publication_receipts",
    "idempotent_reexecution",
    "crash_recovery_supported",
    "database_outside_public_repository",
    "product_host_encryption_isolation",
    "full_integrity_verification",
    "sanitized_inspection",
    "physical_tier_movement_implemented",
    "automatic_retention_implemented",
    "payload_deletion_implemented",
    "deletion_propagation_implemented",
    "storage_pressure_eviction_implemented",
    "backup_restore_implemented",
    "learning_curator_implemented",
    "phase2_memory_store_migrated",
    "friday_product_source_implemented",
    "private_payload_allowed",
    "kernel_key_custody",
    "network_or_cloud_dependency",
}

_REQUIRED_TRUE = {
    "approved_decision_binding",
    "decision_digest_binding",
    "lifecycle_journal_integrity_required",
    "retention_blocker_revalidation",
    "superseded_decision_rejection",
    "non_destructive_source_preservation",
    "host_sealed_opaque_bytes_only",
    "atomic_copy_verify_publish",
    "append_only_transition_intents",
    "append_only_publication_receipts",
    "idempotent_reexecution",
    "crash_recovery_supported",
    "database_outside_public_repository",
    "product_host_encryption_isolation",
    "full_integrity_verification",
    "sanitized_inspection",
    "physical_tier_movement_implemented",
}

_REQUIRED_CONTRACTS = (
    "storage_lifecycle",
    "raw_buffer",
    "retention_lifecycle_journal",
)
_REQUIRED_SCOPE_FIELDS = (
    "product_id",
    "host_instance_id",
    "encryption_domain",
)
_PUBLICATION_PROTOCOL = (
    "append_prepared_intent",
    "copy_host_sealed_bytes_to_temporary_object",
    "verify_temporary_length_and_sha256",
    "atomically_publish_target_object",
    "fsync_target_directory",
    "append_publication_receipt",
    "preserve_source_object",
)


@dataclass(frozen=True)
class CognitiveKernelTierTransitionPolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_contracts: tuple[str, ...]
    store_schema_version: str
    database_format: str
    journal_mode: str
    synchronous: str
    hash_algorithm: str
    required_scope_fields: tuple[str, ...]
    executable_decision_types: tuple[str, ...]
    executable_source_tiers: tuple[str, ...]
    executable_target_tiers: tuple[str, ...]
    publication_protocol: tuple[str, ...]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "tier-transition policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        if require_schema_version(self.version, "version") != "0.9.0":
            raise CognitiveKernelContractError(
                "tier-transition policy version changed"
            )
        if self.phase != "5" or self.milestone != "P5.1d":
            raise CognitiveKernelContractError(
                "tier-transition policy milestone changed"
            )
        if self.status != "runtime_foundation":
            raise CognitiveKernelContractError(
                "tier-transition policy status is invalid"
            )
        if self.required_contracts != _REQUIRED_CONTRACTS:
            raise CognitiveKernelContractError(
                "tier-transition required contracts changed"
            )
        if self.store_schema_version != TIER_TRANSITION_SCHEMA_VERSION:
            raise CognitiveKernelContractError(
                "tier-transition store schema version changed"
            )
        if self.database_format != "sqlite3":
            raise CognitiveKernelContractError(
                "tier-transition database format changed"
            )
        if self.journal_mode != "wal" or self.synchronous != "full":
            raise CognitiveKernelContractError(
                "tier-transition durability settings changed"
            )
        if self.hash_algorithm != "sha256":
            raise CognitiveKernelContractError(
                "tier-transition hash algorithm changed"
            )
        if self.required_scope_fields != _REQUIRED_SCOPE_FIELDS:
            raise CognitiveKernelContractError(
                "tier-transition scope fields changed"
            )
        if set(self.executable_decision_types) != set(
            TIER_TRANSITION_DECISION_TYPES
        ):
            raise CognitiveKernelContractError(
                "tier-transition decision types changed"
            )
        if set(self.executable_source_tiers) != set(
            EXECUTABLE_SOURCE_TIERS
        ):
            raise CognitiveKernelContractError(
                "tier-transition source tiers changed"
            )
        if set(self.executable_target_tiers) != set(
            EXECUTABLE_TARGET_TIERS
        ):
            raise CognitiveKernelContractError(
                "tier-transition target tiers changed"
            )
        if self.publication_protocol != _PUBLICATION_PROTOCOL:
            raise CognitiveKernelContractError(
                "tier-transition publication protocol changed"
            )
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError(
                "tier-transition invariants changed"
            )
        for key in _REQUIRED_TRUE:
            if self.invariants.get(key) is not True:
                raise CognitiveKernelContractError(
                    f"tier-transition invariant {key} must be true"
                )
        for key in _INVARIANT_KEYS - _REQUIRED_TRUE:
            if self.invariants.get(key) is not False:
                raise CognitiveKernelContractError(
                    f"tier-transition invariant {key} must be false"
                )
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError(
                "tier-transition policy created a capability ceiling"
            )
        material = self.material_record()
        if self.digest != canonical_sha256(material):
            raise CognitiveKernelContractError(
                "tier-transition policy digest mismatch"
            )

    def material_record(self) -> dict[str, object]:
        return {
            "cognitive_kernel_tier_transition_policy_schema_version": (
                self.schema_version
            ),
            "policy_id": self.policy_id,
            "version": self.version,
            "phase": self.phase,
            "milestone": self.milestone,
            "status": self.status,
            "required_contracts": list(self.required_contracts),
            "store_schema_version": self.store_schema_version,
            "database_format": self.database_format,
            "journal_mode": self.journal_mode,
            "synchronous": self.synchronous,
            "hash_algorithm": self.hash_algorithm,
            "required_scope_fields": list(self.required_scope_fields),
            "executable_decision_types": list(
                self.executable_decision_types
            ),
            "executable_source_tiers": list(
                self.executable_source_tiers
            ),
            "executable_target_tiers": list(
                self.executable_target_tiers
            ),
            "publication_protocol": list(self.publication_protocol),
            "invariants": dict(self.invariants),
            "capability_ceiling": self.capability_ceiling,
        }


def _sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise CognitiveKernelContractError(f"{field} must be a string list")
    return tuple(value)


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cognitive_kernel_tier_transition_policy(
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelTierTransitionPolicy:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else default_repository_root().resolve(strict=True)
    )
    path = root / "policies" / "cognitive_kernel_tier_transition_policy.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CognitiveKernelContractError(
            "unable to load tier-transition policy"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != _POLICY_KEYS:
        raise CognitiveKernelContractError(
            "tier-transition policy keys changed"
        )
    invariants = raw.get("invariants")
    if not isinstance(invariants, dict) or not all(
        isinstance(key, str) and isinstance(value, bool)
        for key, value in invariants.items()
    ):
        raise CognitiveKernelContractError(
            "tier-transition invariants must be a boolean object"
        )
    policy = CognitiveKernelTierTransitionPolicy(
        schema_version=int(
            raw["cognitive_kernel_tier_transition_policy_schema_version"]
        ),
        policy_id=str(raw["policy_id"]),
        version=str(raw["version"]),
        phase=str(raw["phase"]),
        milestone=str(raw["milestone"]),
        status=str(raw["status"]),
        required_contracts=_sequence(
            raw["required_contracts"], "required_contracts"
        ),
        store_schema_version=str(raw["store_schema_version"]),
        database_format=str(raw["database_format"]),
        journal_mode=str(raw["journal_mode"]),
        synchronous=str(raw["synchronous"]),
        hash_algorithm=str(raw["hash_algorithm"]),
        required_scope_fields=_sequence(
            raw["required_scope_fields"], "required_scope_fields"
        ),
        executable_decision_types=_sequence(
            raw["executable_decision_types"],
            "executable_decision_types",
        ),
        executable_source_tiers=_sequence(
            raw["executable_source_tiers"],
            "executable_source_tiers",
        ),
        executable_target_tiers=_sequence(
            raw["executable_target_tiers"],
            "executable_target_tiers",
        ),
        publication_protocol=_sequence(
            raw["publication_protocol"], "publication_protocol"
        ),
        invariants=dict(invariants),
        capability_ceiling=raw["capability_ceiling"] is True,
        digest="0" * 64,
        source_path=path,
    )
    policy = CognitiveKernelTierTransitionPolicy(
        **{
            **policy.__dict__,
            "digest": canonical_sha256(policy.material_record()),
        }
    )
    policy.validate()
    return policy
