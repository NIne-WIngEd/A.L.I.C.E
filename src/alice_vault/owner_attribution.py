from __future__ import annotations

import copy
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OWNER_IDENTITY_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def default_owner_identity_path(vault_root: Path) -> Path:
    return (
        vault_root.expanduser().resolve()
        / "config"
        / "owner_identity.json"
    )


def _normalize(value: Any) -> str:
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            str(value or "").casefold(),
        )
    )


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = _normalize(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def initialize_owner_identity(
    *,
    vault_root: Path,
    primary_name: str,
    aliases: list[str] | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    primary_name = str(primary_name or "").strip()
    if not primary_name:
        raise ValueError("primary_name may not be empty")

    path = default_owner_identity_path(vault_root)
    if path.exists() and not replace:
        raise FileExistsError(
            "Owner identity already exists. Use --replace only after "
            "reviewing the current private configuration."
        )

    alias_values = _unique(
        [primary_name] + list(aliases or [])
    )
    data = {
        "owner_identity_schema_version": (
            OWNER_IDENTITY_SCHEMA_VERSION
        ),
        "primary_name": primary_name,
        "aliases": alias_values,
        "created_at": _now(),
        "private_config": True,
    }
    _atomic_json(path, data)
    return {
        "owner_identity_path": str(path),
        "primary_name": primary_name,
        "alias_count": len(alias_values),
        "private_config": True,
    }


def load_owner_identity(
    *,
    vault_root: Path,
    require: bool = True,
) -> dict[str, Any] | None:
    vault_root = vault_root.expanduser().resolve(strict=True)
    path = default_owner_identity_path(vault_root)
    if not path.is_file():
        if require:
            raise FileNotFoundError(
                "Private owner identity is missing. Run "
                "scripts/init_owner_identity.py first."
            )
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(data.get("owner_identity_schema_version", -1))
        != OWNER_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported owner-identity schema")

    primary = str(data.get("primary_name", "")).strip()
    aliases = _unique(
        [primary]
        + [
            str(value)
            for value in data.get("aliases", [])
        ]
    )
    if not primary or not aliases:
        raise ValueError("Owner identity contains no usable names")

    return {
        **data,
        "primary_name": primary,
        "aliases": aliases,
        "owner_identity_path": str(path),
    }


_SELF_RECORD_HINTS = (
    "resume",
    "curriculum vitae",
    " cv ",
    "portfolio",
    "personal profile",
    "professional profile",
    "personal statement",
)

_ACCOUNT_RECORD_HINTS = (
    "work html",
    "research html",
    "profile html",
)


def _contains_alias(
    value: str,
    aliases: list[str],
) -> bool:
    normalized = f" {_normalize(value)} "
    for alias in aliases:
        candidate = _normalize(alias)
        if candidate and f" {candidate} " in normalized:
            return True
    return False


def _has_hint(
    value: str,
    hints: tuple[str, ...],
) -> str:
    normalized = f" {_normalize(value)} "
    for hint in hints:
        normalized_hint = f" {_normalize(hint)} "
        if normalized_hint in normalized:
            return hint.strip()
    return ""


def classify_owner_relation(
    *,
    evidence: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    aliases = list(identity["aliases"])
    provenance = list(evidence.get("provenance", []))

    metadata_parts: list[str] = []
    for item in provenance:
        metadata_parts.extend(
            [
                str(item.get("filename", "")),
                str(item.get("original_relative_path", "")),
                str(item.get("source_bucket", "")),
            ]
        )
    metadata = " ".join(metadata_parts)
    context_text = str(
        evidence.get("context_text", "")
    )

    alias_in_metadata = _contains_alias(
        metadata,
        aliases,
    )
    alias_in_text = _contains_alias(
        context_text,
        aliases,
    )
    self_record_hint = _has_hint(
        metadata,
        _SELF_RECORD_HINTS,
    )
    account_record_hint = _has_hint(
        metadata,
        _ACCOUNT_RECORD_HINTS,
    )

    if alias_in_metadata and self_record_hint:
        return {
            "owner_relation": "owner_self_record",
            "owner_relation_confidence": "high",
            "owner_relation_basis": (
                "owner alias appears in provenance metadata and "
                f"source type resembles {self_record_hint!r}"
            ),
        }

    if alias_in_text and self_record_hint:
        return {
            "owner_relation": "owner_self_record",
            "owner_relation_confidence": "high",
            "owner_relation_basis": (
                "owner alias appears in evidence text and source "
                f"type resembles {self_record_hint!r}"
            ),
        }

    if alias_in_metadata or alias_in_text:
        return {
            "owner_relation": "owner_related_record",
            "owner_relation_confidence": "medium",
            "owner_relation_basis": (
                "owner alias appears in source metadata or evidence text"
            ),
        }

    if account_record_hint:
        return {
            "owner_relation": "owner_account_record_unverified",
            "owner_relation_confidence": "low",
            "owner_relation_basis": (
                "source resembles an account-export record but contains "
                "no deterministic owner-name match"
            ),
        }

    return {
        "owner_relation": "unknown",
        "owner_relation_confidence": "none",
        "owner_relation_basis": (
            "no deterministic owner attribution signal"
        ),
    }


def annotate_context_owner_relation(
    *,
    vault_root: Path,
    context_package: dict[str, Any],
    require_identity: bool = True,
) -> dict[str, Any]:
    """Attach deterministic, private owner-attribution metadata.

    This does not change retrieval ranking or citations. It only tells the
    response layer whether a retrieved record can safely be treated as a
    self-record belonging to the vault owner.
    """
    from .grounded_context import _fingerprint

    vault_root = vault_root.expanduser().resolve(strict=True)
    identity = load_owner_identity(
        vault_root=vault_root,
        require=require_identity,
    )
    package = copy.deepcopy(context_package)

    if identity is None:
        package["owner_identity_context"] = {
            "available": False,
            "private_config": True,
        }
        return package

    evidence = list(package.get("evidence", []))
    counts: dict[str, int] = {}
    for item in evidence:
        relation = classify_owner_relation(
            evidence=item,
            identity=identity,
        )
        item.update(relation)
        key = relation["owner_relation"]
        counts[key] = counts.get(key, 0) + 1

    package["evidence"] = evidence
    package["owner_identity_context"] = {
        "available": True,
        "owner_primary_name": identity[
            "primary_name"
        ],
        "owner_alias_count": len(
            identity["aliases"]
        ),
        "relation_counts": counts,
        "private_config": True,
        "attribution_method": (
            "deterministic_alias_and_record_type_v1"
        ),
    }
    package["package_fingerprint"] = _fingerprint(
        package
    )
    return package
