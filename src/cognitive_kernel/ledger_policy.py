"""Loader for the P5.1a compact experience-ledger policy."""

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

_POLICY_KEYS = {
    "cognitive_kernel_experience_ledger_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "required_contracts",
    "database_format",
    "ledger_schema_version",
    "journal_mode",
    "synchronous",
    "hash_algorithm",
    "required_scope_fields",
    "invariants",
    "capability_ceiling",
}
_INVARIANT_KEYS = {
    "logically_append_only",
    "deterministic_sequence_numbers",
    "tamper_evident_hash_chain",
    "duplicate_event_rejection",
    "database_outside_public_repository",
    "metadata_only_event_storage",
    "product_host_encryption_isolation",
    "atomic_transactions",
    "full_integrity_verification",
    "sanitized_inspection",
    "raw_buffer_implemented",
    "blob_store_implemented",
    "payload_deduplication_implemented",
    "automated_retention_implemented",
    "backup_restore_implemented",
    "phase2_memory_store_migrated",
    "friday_product_source_implemented",
    "private_payload_allowed",
}
_REQUIRED_CONTRACTS = ("experience_event", "storage_lifecycle")
_REQUIRED_SCOPE_FIELDS = (
    "product_id",
    "host_instance_id",
    "encryption_domain",
)


@dataclass(frozen=True)
class CognitiveKernelExperienceLedgerPolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_contracts: tuple[str, ...]
    database_format: str
    ledger_schema_version: str
    journal_mode: str
    synchronous: str
    hash_algorithm: str
    required_scope_fields: tuple[str, ...]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "experience-ledger policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        require_schema_version(self.version, "version")
        if self.version != "0.6.0":
            raise CognitiveKernelContractError(
                "experience-ledger policy version changed"
            )
        if self.phase != "5" or self.milestone != "P5.1a":
            raise CognitiveKernelContractError(
                "experience-ledger policy milestone changed"
            )
        if self.status != "runtime_foundation":
            raise CognitiveKernelContractError(
                "experience-ledger policy status is invalid"
            )
        if self.required_contracts != _REQUIRED_CONTRACTS:
            raise CognitiveKernelContractError(
                "experience-ledger required contracts changed"
            )
        if self.database_format != "sqlite3":
            raise CognitiveKernelContractError(
                "experience-ledger database format changed"
            )
        if self.ledger_schema_version != "1.0.0":
            raise CognitiveKernelContractError(
                "experience-ledger schema version changed"
            )
        if self.journal_mode != "wal" or self.synchronous != "full":
            raise CognitiveKernelContractError(
                "experience-ledger durability settings changed"
            )
        if self.hash_algorithm != "sha256":
            raise CognitiveKernelContractError(
                "experience-ledger hash algorithm changed"
            )
        if self.required_scope_fields != _REQUIRED_SCOPE_FIELDS:
            raise CognitiveKernelContractError(
                "experience-ledger scope fields changed"
            )
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError(
                "experience-ledger invariants changed"
            )
        required_true = {
            "logically_append_only",
            "deterministic_sequence_numbers",
            "tamper_evident_hash_chain",
            "duplicate_event_rejection",
            "database_outside_public_repository",
            "metadata_only_event_storage",
            "product_host_encryption_isolation",
            "atomic_transactions",
            "full_integrity_verification",
            "sanitized_inspection",
        }
        for key in required_true:
            if self.invariants.get(key) is not True:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be true"
                )
        for key in _INVARIANT_KEYS - required_true:
            if self.invariants.get(key) is not False:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be false"
                )
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError(
                "experience-ledger policy may not be a capability ceiling"
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


def _sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CognitiveKernelContractError(
            f"{field} must be a non-empty list"
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


def load_cognitive_kernel_experience_ledger_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelExperienceLedgerPolicy:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = (
        Path(path).resolve()
        if path is not None
        else root
        / "policies"
        / "cognitive_kernel_experience_ledger_policy.json"
    )
    payload = _read_json(source)
    _exact_keys(payload, _POLICY_KEYS, "experience-ledger policy")
    invariants = _require_dict(payload["invariants"], "invariants")
    _exact_keys(invariants, _INVARIANT_KEYS, "experience-ledger invariants")
    if any(not isinstance(value, bool) for value in invariants.values()):
        raise CognitiveKernelContractError(
            "experience-ledger invariants must be booleans"
        )
    policy = CognitiveKernelExperienceLedgerPolicy(
        schema_version=int(
            payload["cognitive_kernel_experience_ledger_policy_schema_version"]
        ),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(payload["milestone"], "milestone", maximum=32),
        status=require_identifier(payload["status"], "status"),
        required_contracts=_sequence(
            payload["required_contracts"], "required_contracts"
        ),
        database_format=require_identifier(
            payload["database_format"], "database_format"
        ),
        ledger_schema_version=require_schema_version(
            payload["ledger_schema_version"], "ledger_schema_version"
        ),
        journal_mode=require_identifier(payload["journal_mode"], "journal_mode"),
        synchronous=require_identifier(payload["synchronous"], "synchronous"),
        hash_algorithm=require_identifier(
            payload["hash_algorithm"], "hash_algorithm"
        ),
        required_scope_fields=_sequence(
            payload["required_scope_fields"], "required_scope_fields"
        ),
        invariants={str(key): bool(value) for key, value in invariants.items()},
        capability_ceiling=bool(payload["capability_ceiling"]),
        digest=canonical_sha256(payload),
        source_path=source,
    )
    policy.validate()
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
    storage = _read_json(root / "policies" / "storage_lifecycle_policy.json")
    if storage.get("permanent_compact_event_ledger") is not True:
        raise CognitiveKernelContractError(
            "storage policy does not require a permanent compact ledger"
        )
    addressing = _require_dict(storage.get("content_addressing"), "content_addressing")
    if addressing.get("algorithm") != "sha256":
        raise CognitiveKernelContractError(
            "storage policy hash algorithm changed"
        )
    if addressing.get("deduplication_scope") != (
        "host_instance_and_encryption_domain"
    ):
        raise CognitiveKernelContractError(
            "storage policy scope changed"
        )
    if addressing.get("cross_host_deduplication_allowed") is not False:
        raise CognitiveKernelContractError(
            "storage policy permits cross-host deduplication"
        )
    if addressing.get("logical_metadata_must_remain_distinct") is not True:
        raise CognitiveKernelContractError(
            "storage policy does not preserve distinct metadata events"
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
