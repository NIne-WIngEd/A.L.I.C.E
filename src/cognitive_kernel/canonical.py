"""Canonical serialization and validation helpers for kernel contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Iterable

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,255}$")
_SCOPE_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_SCHEMA_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){0,2}$")


class CognitiveKernelContractError(ValueError):
    """Raised when a kernel contract is malformed or tampered."""


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON and reject non-finite numbers."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CognitiveKernelContractError(
            "value is not canonical JSON material"
        ) from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_text(
    value: object,
    field: str,
    *,
    maximum: int = 4096,
    allow_newlines: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CognitiveKernelContractError(f"{field} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > maximum or "\x00" in normalized:
        raise CognitiveKernelContractError(f"{field} is invalid")
    if not allow_newlines and ("\r" in normalized or "\n" in normalized):
        raise CognitiveKernelContractError(f"{field} may not contain newlines")
    return normalized


def require_identifier(value: object, field: str) -> str:
    normalized = require_text(value, field, maximum=256).lower()
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise CognitiveKernelContractError(f"{field} is not a valid identifier")
    return normalized


def require_scope_identifier(value: object, field: str) -> str:
    normalized = require_text(value, field, maximum=128).lower()
    if _SCOPE_IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise CognitiveKernelContractError(
            f"{field} is not a valid scope identifier"
        )
    return normalized


def require_schema_version(value: object, field: str = "schema_version") -> str:
    normalized = require_text(value, field, maximum=32)
    if _SCHEMA_VERSION_PATTERN.fullmatch(normalized) is None:
        raise CognitiveKernelContractError(f"{field} is invalid")
    return normalized


def require_sha256(value: object, field: str) -> str:
    normalized = require_text(value, field, maximum=64).lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise CognitiveKernelContractError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return normalized


def normalize_timestamp(value: object, field: str = "timestamp") -> str:
    normalized = require_text(value, field, maximum=64)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CognitiveKernelContractError(
            f"{field} must be valid ISO-8601 text"
        ) from exc
    if parsed.tzinfo is None:
        raise CognitiveKernelContractError(
            f"{field} must include a timezone offset"
        )
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def require_confidence(value: object | None, field: str = "confidence") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveKernelContractError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise CognitiveKernelContractError(
            f"{field} must be finite and between 0 and 1"
        )
    return normalized


def normalize_identifier_sequence(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise CognitiveKernelContractError(f"{field} must be a sequence")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = require_identifier(value, field)
        if item in seen:
            raise CognitiveKernelContractError(
                f"{field} may not contain duplicates"
            )
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)
