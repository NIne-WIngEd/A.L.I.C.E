from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ParserSpec:
    parser_id: str
    family: str
    enabled: bool
    extensions: tuple[str, ...]
    dependency: str
    max_file_bytes: int
    timeout_seconds: int
    max_output_chars: int
    limits: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParserSpec":
        return cls(
            parser_id=str(value["parser_id"]),
            family=str(value["family"]).strip().lower(),
            enabled=bool(value["enabled"]),
            extensions=tuple(
                str(item).strip().lower()
                for item in value.get("extensions", [])
            ),
            dependency=str(value["dependency"]),
            max_file_bytes=int(value["max_file_bytes"]),
            timeout_seconds=int(value["timeout_seconds"]),
            max_output_chars=int(value["max_output_chars"]),
            limits=dict(value.get("limits", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_id": self.parser_id,
            "family": self.family,
            "enabled": self.enabled,
            "extensions": list(self.extensions),
            "dependency": self.dependency,
            "max_file_bytes": self.max_file_bytes,
            "timeout_seconds": self.timeout_seconds,
            "max_output_chars": self.max_output_chars,
            "limits": self.limits,
        }


@dataclass(frozen=True)
class ParserRegistry:
    registry_id: str
    default_policy: dict[str, Any]
    parsers: tuple[ParserSpec, ...]
    digest: str
    source_path: Path

    def by_family(self) -> dict[str, ParserSpec]:
        return {parser.family: parser for parser in self.parsers}

    def select(self, family: str, source_path: Path) -> ParserSpec:
        family = family.strip().lower()
        parser = self.by_family().get(family)
        if parser is None:
            raise ValueError(f"No parser is registered for family {family!r}")
        if not parser.enabled:
            raise ValueError(
                f"Parser family {family!r} is disabled: {parser.parser_id}"
            )

        extension = source_path.suffix.lower()
        if extension not in parser.extensions:
            raise ValueError(
                f"Extension {extension!r} is not allowed for family "
                f"{family!r}; expected one of {parser.extensions}"
            )
        return parser


def default_registry_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "parser_registry.json"
    )


def load_registry(path: Path | None = None) -> ParserRegistry:
    source = (path or default_registry_path()).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))

    if int(data.get("registry_schema_version", -1)) != REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported parser-registry schema version")

    parsers = tuple(
        ParserSpec.from_dict(item) for item in data.get("parsers", [])
    )
    if not parsers:
        raise ValueError("Parser registry contains no parsers")

    parser_ids = [parser.parser_id for parser in parsers]
    families = [parser.family for parser in parsers]
    if len(parser_ids) != len(set(parser_ids)):
        raise ValueError("Parser IDs must be unique")
    if len(families) != len(set(families)):
        raise ValueError("Parser families must be unique")

    for parser in parsers:
        if parser.enabled:
            if not parser.extensions:
                raise ValueError(
                    f"Enabled parser {parser.parser_id} has no extensions"
                )
            if parser.max_file_bytes <= 0:
                raise ValueError(
                    f"Enabled parser {parser.parser_id} has invalid size limit"
                )
            if parser.timeout_seconds <= 0:
                raise ValueError(
                    f"Enabled parser {parser.parser_id} has invalid timeout"
                )
            if parser.max_output_chars <= 0:
                raise ValueError(
                    f"Enabled parser {parser.parser_id} has invalid output cap"
                )

    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ParserRegistry(
        registry_id=str(data["registry_id"]),
        default_policy=dict(data.get("default_policy", {})),
        parsers=parsers,
        digest=hashlib.sha256(canonical).hexdigest(),
        source_path=source,
    )
