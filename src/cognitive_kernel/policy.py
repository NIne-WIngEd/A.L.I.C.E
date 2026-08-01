"""Loader for the Phase 5 Cognitive Kernel foundation policy."""

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
from .contracts import (
    IDENTITY_LAYERS,
    PRIVATE_DIRECTIVE_CODES,
    PRODUCT_IDS,
    PROVENANCE_TYPES,
    RETENTION_CLASSES,
    STORAGE_TIERS,
)

_POLICY_KEYS = {
    "cognitive_kernel_foundation_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "shared_kernel_id",
    "required_products",
    "required_scope_fields",
    "allowed_provenance_types",
    "allowed_identity_layers",
    "allowed_private_directive_codes",
    "allowed_storage_tiers",
    "allowed_retention_classes",
    "boundaries",
    "capability_ceiling",
}
_BOUNDARY_KEYS = {
    "metadata_only_contracts",
    "private_payload_allowed",
    "cross_host_deduplication_allowed",
    "friday_product_source_allowed",
    "persistent_store_implemented",
    "raw_buffer_implemented",
    "complete_ui_implemented",
    "autonomous_learning_implemented",
}
_REQUIRED_SCOPE_FIELDS = (
    "product_id",
    "host_instance_id",
    "schema_version",
    "encryption_domain",
    "provenance",
    "content_digest",
    "retention_class",
    "storage_tier",
    "deletion_lineage",
)


@dataclass(frozen=True)
class CognitiveKernelFoundationPolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    shared_kernel_id: str
    required_products: tuple[str, ...]
    required_scope_fields: tuple[str, ...]
    allowed_provenance_types: tuple[str, ...]
    allowed_identity_layers: tuple[str, ...]
    allowed_private_directive_codes: tuple[str, ...]
    allowed_storage_tiers: tuple[str, ...]
    allowed_retention_classes: tuple[str, ...]
    boundaries: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "foundation policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        require_schema_version(self.version, "version")
        if self.phase != "5" or self.milestone != "P5.0b":
            raise CognitiveKernelContractError(
                "foundation policy must bind Phase 5 milestone P5.0b"
            )
        if self.status != "contract_foundation":
            raise CognitiveKernelContractError(
                "foundation policy status is invalid"
            )
        if self.shared_kernel_id != "personal-cognitive-kernel":
            raise CognitiveKernelContractError(
                "shared kernel identity changed"
            )
        if set(self.required_products) != PRODUCT_IDS:
            raise CognitiveKernelContractError(
                "required product identities changed"
            )
        if tuple(self.required_scope_fields) != _REQUIRED_SCOPE_FIELDS:
            raise CognitiveKernelContractError(
                "required kernel scope fields changed"
            )
        if set(self.allowed_provenance_types) != PROVENANCE_TYPES:
            raise CognitiveKernelContractError(
                "provenance vocabulary changed"
            )
        if set(self.allowed_identity_layers) != IDENTITY_LAYERS:
            raise CognitiveKernelContractError(
                "identity-layer vocabulary changed"
            )
        if (
            set(self.allowed_private_directive_codes)
            != PRIVATE_DIRECTIVE_CODES
        ):
            raise CognitiveKernelContractError(
                "private directive-code vocabulary changed"
            )
        if set(self.allowed_storage_tiers) != STORAGE_TIERS:
            raise CognitiveKernelContractError(
                "storage-tier vocabulary changed"
            )
        if set(self.allowed_retention_classes) != RETENTION_CLASSES:
            raise CognitiveKernelContractError(
                "retention-class vocabulary changed"
            )
        if set(self.boundaries) != _BOUNDARY_KEYS:
            raise CognitiveKernelContractError(
                "foundation policy boundaries changed"
            )
        required_false = {
            "private_payload_allowed",
            "cross_host_deduplication_allowed",
            "friday_product_source_allowed",
            "persistent_store_implemented",
            "raw_buffer_implemented",
            "complete_ui_implemented",
            "autonomous_learning_implemented",
        }
        if self.boundaries.get("metadata_only_contracts") is not True:
            raise CognitiveKernelContractError(
                "foundation must remain metadata-only"
            )
        for key in required_false:
            if self.boundaries.get(key) is not False:
                raise CognitiveKernelContractError(
                    f"foundation boundary {key!r} must be false"
                )
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError(
                "foundation policy may not be a capability ceiling"
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
    *,
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise CognitiveKernelContractError(
            f"{field} keys changed; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CognitiveKernelContractError(
            f"{field} must be a non-empty list"
        )
    normalized = tuple(
        require_text(item, field, maximum=256)
        for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise CognitiveKernelContractError(
            f"{field} may not contain duplicates"
        )
    return normalized


def load_cognitive_kernel_foundation_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelFoundationPolicy:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = (
        Path(path).resolve()
        if path is not None
        else root / "policies" / "cognitive_kernel_foundation_policy.json"
    )
    payload = _read_json(source)
    _exact_keys(payload, _POLICY_KEYS, field="foundation policy")

    boundaries = payload["boundaries"]
    if not isinstance(boundaries, dict):
        raise CognitiveKernelContractError(
            "foundation boundaries must be an object"
        )
    _exact_keys(boundaries, _BOUNDARY_KEYS, field="foundation boundaries")
    if any(not isinstance(value, bool) for value in boundaries.values()):
        raise CognitiveKernelContractError(
            "foundation boundaries must be booleans"
        )

    policy = CognitiveKernelFoundationPolicy(
        schema_version=int(
            payload["cognitive_kernel_foundation_policy_schema_version"]
        ),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(
            payload["milestone"], "milestone", maximum=32
        ),
        status=require_identifier(payload["status"], "status"),
        shared_kernel_id=require_identifier(
            payload["shared_kernel_id"], "shared_kernel_id"
        ),
        required_products=_text_sequence(
            payload["required_products"], "required_products"
        ),
        required_scope_fields=_text_sequence(
            payload["required_scope_fields"],
            "required_scope_fields",
        ),
        allowed_provenance_types=_text_sequence(
            payload["allowed_provenance_types"],
            "allowed_provenance_types",
        ),
        allowed_identity_layers=_text_sequence(
            payload["allowed_identity_layers"],
            "allowed_identity_layers",
        ),
        allowed_private_directive_codes=_text_sequence(
            payload["allowed_private_directive_codes"],
            "allowed_private_directive_codes",
        ),
        allowed_storage_tiers=_text_sequence(
            payload["allowed_storage_tiers"],
            "allowed_storage_tiers",
        ),
        allowed_retention_classes=_text_sequence(
            payload["allowed_retention_classes"],
            "allowed_retention_classes",
        ),
        boundaries={
            str(key): bool(value)
            for key, value in boundaries.items()
        },
        capability_ceiling=bool(payload["capability_ceiling"]),
        digest=canonical_sha256(payload),
        source_path=source,
    )
    policy.validate()

    product_lines = _read_json(root / "policies" / "product_lines.json")
    shared_kernel = product_lines.get("shared_kernel")
    if not isinstance(shared_kernel, dict):
        raise CognitiveKernelContractError(
            "product manifest shared_kernel is invalid"
        )
    if shared_kernel.get("id") != policy.shared_kernel_id:
        raise CognitiveKernelContractError(
            "foundation and product manifests disagree on kernel identity"
        )
    if tuple(shared_kernel.get("required_scopes", ())) != (
        policy.required_scope_fields
    ):
        raise CognitiveKernelContractError(
            "foundation and product manifests disagree on scope fields"
        )
    products = product_lines.get("products")
    if not isinstance(products, dict):
        raise CognitiveKernelContractError(
            "product manifest products are invalid"
        )
    if set(products) != set(policy.required_products):
        raise CognitiveKernelContractError(
            "foundation and product manifests disagree on products"
        )

    custody = _read_json(
        root / "policies" / "private_companion_custody.json"
    )
    if set(custody.get("provenance_types", ())) != set(
        policy.allowed_provenance_types
    ):
        raise CognitiveKernelContractError(
            "foundation and custody policies disagree on provenance"
        )
    if set(custody.get("directive_codes", ())) != set(
        policy.allowed_private_directive_codes
    ):
        raise CognitiveKernelContractError(
            "foundation and custody policies disagree on directive codes"
        )

    identity = _read_json(
        root / "policies" / "alice_clone_identity_policy.json"
    )
    if set(identity.get("required_identity_layers", ())) != set(
        policy.allowed_identity_layers
    ):
        raise CognitiveKernelContractError(
            "foundation and identity policies disagree on identity layers"
        )

    lifecycle = _read_json(
        root / "policies" / "storage_lifecycle_policy.json"
    )
    tiers = lifecycle.get("tiers")
    retention_classes = lifecycle.get("retention_classes")
    if not isinstance(tiers, dict) or set(tiers) != set(
        policy.allowed_storage_tiers
    ):
        raise CognitiveKernelContractError(
            "foundation and lifecycle policies disagree on storage tiers"
        )
    if (
        not isinstance(retention_classes, dict)
        or set(retention_classes)
        != set(policy.allowed_retention_classes)
    ):
        raise CognitiveKernelContractError(
            "foundation and lifecycle policies disagree on retention classes"
        )
    return policy
