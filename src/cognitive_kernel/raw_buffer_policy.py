"""Loader for the P5.1b raw-buffer and content-store policy."""

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
    "cognitive_kernel_raw_buffer_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "required_contracts",
    "payload_mode",
    "content_digest_algorithm",
    "deduplication_scope",
    "required_scope_fields",
    "invariants",
    "capability_ceiling",
}
_INVARIANT_KEYS = {
    "persistent_ledger_implemented",
    "policy_bounded_raw_buffer",
    "host_sealed_opaque_payloads",
    "content_addressed_objects",
    "same_scope_physical_deduplication",
    "logical_references_remain_distinct",
    "cross_host_deduplication_prohibited",
    "atomic_object_publication",
    "transactional_reference_metadata",
    "database_and_objects_outside_public_repository",
    "payload_integrity_verification",
    "sanitized_inspection",
    "storage_accounting",
    "kernel_key_custody",
    "plaintext_payload_requirement",
    "automatic_expiry_implemented",
    "tier_movement_implemented",
    "deletion_propagation_implemented",
    "storage_pressure_eviction_implemented",
    "backup_restore_implemented",
    "friday_product_source_implemented",
    "payload_in_source_or_contracts_allowed",
}
_REQUIRED_TRUE = {
    "persistent_ledger_implemented",
    "policy_bounded_raw_buffer",
    "host_sealed_opaque_payloads",
    "content_addressed_objects",
    "same_scope_physical_deduplication",
    "logical_references_remain_distinct",
    "cross_host_deduplication_prohibited",
    "atomic_object_publication",
    "transactional_reference_metadata",
    "database_and_objects_outside_public_repository",
    "payload_integrity_verification",
    "sanitized_inspection",
    "storage_accounting",
}
_REQUIRED_CONTRACTS = ("experience_event", "storage_lifecycle")
_REQUIRED_SCOPE_FIELDS = (
    "product_id",
    "host_instance_id",
    "encryption_domain",
)


@dataclass(frozen=True)
class CognitiveKernelRawBufferPolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_contracts: tuple[str, ...]
    payload_mode: str
    content_digest_algorithm: str
    deduplication_scope: str
    required_scope_fields: tuple[str, ...]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "raw-buffer policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        if require_schema_version(self.version, "version") != "0.7.0":
            raise CognitiveKernelContractError("raw-buffer policy version changed")
        if self.phase != "5" or self.milestone != "P5.1b":
            raise CognitiveKernelContractError("raw-buffer milestone changed")
        if self.status != "runtime_foundation":
            raise CognitiveKernelContractError("raw-buffer status is invalid")
        if self.required_contracts != _REQUIRED_CONTRACTS:
            raise CognitiveKernelContractError(
                "raw-buffer required contracts changed"
            )
        if self.payload_mode != "host_sealed_opaque_bytes":
            raise CognitiveKernelContractError("raw-buffer payload mode changed")
        if self.content_digest_algorithm != "sha256":
            raise CognitiveKernelContractError("raw-buffer digest changed")
        if self.deduplication_scope != "host_instance_and_encryption_domain":
            raise CognitiveKernelContractError(
                "raw-buffer deduplication scope changed"
            )
        if self.required_scope_fields != _REQUIRED_SCOPE_FIELDS:
            raise CognitiveKernelContractError("raw-buffer scope fields changed")
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError("raw-buffer invariants changed")
        for key in _REQUIRED_TRUE:
            if self.invariants.get(key) is not True:
                raise CognitiveKernelContractError(
                    f"raw-buffer invariant {key} must be true"
                )
        for key in _INVARIANT_KEYS - _REQUIRED_TRUE:
            if self.invariants.get(key) is not False:
                raise CognitiveKernelContractError(
                    f"raw-buffer invariant {key} must be false"
                )
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError(
                "raw-buffer policy may not be a capability ceiling"
            )


def _sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CognitiveKernelContractError(f"{field} must be a non-empty list")
    result = tuple(require_text(item, field, maximum=128) for item in value)
    if len(set(result)) != len(result):
        raise CognitiveKernelContractError(f"{field} may not contain duplicates")
    return result


def load_cognitive_kernel_raw_buffer_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelRawBufferPolicy:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = (
        Path(path).resolve()
        if path is not None
        else root / "policies" / "cognitive_kernel_raw_buffer_policy.json"
    )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CognitiveKernelContractError(
            f"could not load raw-buffer policy: {source}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _POLICY_KEYS:
        raise CognitiveKernelContractError("raw-buffer policy keys changed")
    invariants = payload.get("invariants")
    if not isinstance(invariants, dict) or set(invariants) != _INVARIANT_KEYS:
        raise CognitiveKernelContractError("raw-buffer invariant keys changed")
    if any(not isinstance(value, bool) for value in invariants.values()):
        raise CognitiveKernelContractError(
            "raw-buffer invariants must be booleans"
        )
    policy = CognitiveKernelRawBufferPolicy(
        schema_version=int(
            payload["cognitive_kernel_raw_buffer_policy_schema_version"]
        ),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(payload["milestone"], "milestone", maximum=32),
        status=require_identifier(payload["status"], "status"),
        required_contracts=_sequence(
            payload["required_contracts"], "required_contracts"
        ),
        payload_mode=require_identifier(payload["payload_mode"], "payload_mode"),
        content_digest_algorithm=require_identifier(
            payload["content_digest_algorithm"], "content_digest_algorithm"
        ),
        deduplication_scope=require_identifier(
            payload["deduplication_scope"], "deduplication_scope"
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
    return policy
