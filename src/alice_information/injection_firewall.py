"""Deterministic retrieved-content injection firewall for Phase 4 P4.3."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import unquote

from .contracts import (
    InformationContractError,
    InformationSourceDocument,
    canonicalize_public_url,
    sha256_text,
)
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy

INJECTION_VERDICTS = ("clear", "blocked")
INJECTION_SEVERITIES = ("critical",)
INJECTION_LOCATIONS = ("title", "url", "content")

_ERROR_MESSAGES = {
    "inspection_limit_exceeded": "Retrieved source exceeded the deterministic inspection limit.",
    "inspection_policy_invalid": "Retrieved-source inspection policy validation failed.",
    "source_binding_invalid": "Retrieved-source inspection binding validation failed.",
    "source_blocked": "Retrieved source was blocked by the injection firewall.",
}


class InformationInjectionFirewallError(InformationContractError):
    """Sanitized P4.3 firewall failure with an approved code."""

    def __init__(self, code: str):
        if code not in _ERROR_MESSAGES:
            code = "inspection_policy_invalid"
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


def injection_failure(code: str) -> InformationInjectionFirewallError:
    return InformationInjectionFirewallError(code)


@dataclass(frozen=True)
class InformationInjectionFinding:
    """Metadata-only finding; no raw source excerpt is retained."""

    code: str
    severity: str
    location: str
    line_number: int
    normalized_line_sha256: str

    def validate(self, *, policy: InformationInjectionFirewallPolicy) -> None:
        policy.validate()
        if self.code not in policy.critical_finding_codes:
            raise injection_failure("source_binding_invalid")
        if self.severity not in INJECTION_SEVERITIES:
            raise injection_failure("source_binding_invalid")
        if self.location not in INJECTION_LOCATIONS:
            raise injection_failure("source_binding_invalid")
        if not isinstance(self.line_number, int) or isinstance(self.line_number, bool):
            raise injection_failure("source_binding_invalid")
        if self.line_number < 1:
            raise injection_failure("source_binding_invalid")
        _require_digest(self.normalized_line_sha256)


@dataclass(frozen=True)
class InformationSourceInspection:
    """Digest-bound deterministic inspection result for one source version."""

    source_id: str
    canonical_url: str
    source_content_sha256: str
    source_metadata_sha256: str
    policy_version: str
    verdict: str
    findings: tuple[InformationInjectionFinding, ...]
    finding_codes: tuple[str, ...]
    detection_view_sha256: str
    inspected_characters: int
    inspected_lines: int

    def validate(
        self,
        *,
        source: InformationSourceDocument,
        policy: InformationInjectionFirewallPolicy,
    ) -> None:
        source.validate()
        policy.validate()
        if self.source_id != source.source_id:
            raise injection_failure("source_binding_invalid")
        if self.canonical_url != source.canonical_url:
            raise injection_failure("source_binding_invalid")
        if self.source_content_sha256 != source.content_sha256:
            raise injection_failure("source_binding_invalid")
        if self.source_metadata_sha256 != _source_metadata_sha256(source):
            raise injection_failure("source_binding_invalid")
        if self.policy_version != policy.version:
            raise injection_failure("source_binding_invalid")
        if self.verdict not in INJECTION_VERDICTS:
            raise injection_failure("source_binding_invalid")
        _require_digest(self.detection_view_sha256)
        if not isinstance(self.inspected_characters, int) or self.inspected_characters < 0:
            raise injection_failure("source_binding_invalid")
        if not isinstance(self.inspected_lines, int) or self.inspected_lines < 1:
            raise injection_failure("source_binding_invalid")
        if self.inspected_characters > policy.max_source_characters:
            raise injection_failure("source_binding_invalid")
        if self.inspected_lines > policy.max_source_lines:
            raise injection_failure("source_binding_invalid")
        if len(self.findings) > policy.max_findings:
            raise injection_failure("source_binding_invalid")
        seen: set[tuple[str, str, int, str]] = set()
        derived_codes: list[str] = []
        for finding in self.findings:
            finding.validate(policy=policy)
            key = (
                finding.code,
                finding.location,
                finding.line_number,
                finding.normalized_line_sha256,
            )
            if key in seen:
                raise injection_failure("source_binding_invalid")
            seen.add(key)
            if finding.code not in derived_codes:
                derived_codes.append(finding.code)
        if tuple(derived_codes) != self.finding_codes:
            raise injection_failure("source_binding_invalid")
        if self.verdict == "clear" and self.findings:
            raise injection_failure("source_binding_invalid")
        if self.verdict == "blocked" and not self.findings:
            raise injection_failure("source_binding_invalid")
        expected = _derive_inspection(source, policy)
        if self != expected:
            raise injection_failure("source_binding_invalid")


@dataclass(frozen=True)
class InformationInspectedSource:
    """Only source wrapper eligible for future model-facing web grounding."""

    source: InformationSourceDocument
    inspection: InformationSourceInspection

    def validate(self, *, policy: InformationInjectionFirewallPolicy) -> None:
        self.inspection.validate(source=self.source, policy=policy)

    def render_for_model(self, *, policy: InformationInjectionFirewallPolicy) -> str:
        """Render only an exact, clear, digest-bound source inspection."""

        self.validate(policy=policy)
        if self.inspection.verdict != "clear" or not policy.clear_sources_renderable:
            raise injection_failure("source_blocked")
        if policy.flagged_sources_renderable:
            raise injection_failure("inspection_policy_invalid")
        rendered = self.source.render_for_model()
        inspection_boundary = (
            f"ALICE-INJECTION-INSPECTION-{self.inspection.detection_view_sha256.upper()}"
        )
        return "\n".join(
            (
                f"BEGIN VERIFIED SOURCE INSPECTION {inspection_boundary}",
                f"Inspection policy: {policy.policy_name}@{policy.version}",
                "Injection verdict: clear",
                f"Source content SHA-256: {self.source.content_sha256}",
                rendered,
                f"END VERIFIED SOURCE INSPECTION {inspection_boundary}",
            )
        )


@dataclass(frozen=True)
class _PatternRule:
    code: str
    pattern: re.Pattern[str]


_FLAGS = re.IGNORECASE
_RULES = (
    _PatternRule(
        "boundary_collision_attempt",
        re.compile(
            r"(?:BEGIN|END)\s+(?:UNTRUSTED\s+EXTERNAL\s+SOURCE|SOURCE\s+CONTENT|VERIFIED\s+SOURCE\s+INSPECTION|VERIFIED\s+SOURCE\s+FRESHNESS)|ALICE-(?:EXTERNAL-SOURCE|INJECTION-INSPECTION|FRESHNESS-ASSESSMENT)-",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "role_marker_instruction",
        re.compile(
            r"(?:^|\s|\[)(?:<\|\s*)?(?:system|developer|assistant)(?:\s*\|>)?\s*(?:message|prompt)?\s*[:\]]|^\s*#{1,6}\s*(?:system|developer|assistant)\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "policy_override_instruction",
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|system|developer)\s+(?:instructions?|messages?|rules?)\b|\boverride\s+(?:the\s+)?(?:system|developer|policy|rules?|guardrails?)\b|\bfollow\s+(?:these|the following|my)\s+instructions?\s+instead\b|\bnew\s+system\s+prompt\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "permission_laundering_instruction",
        re.compile(
            r"\b(?:you\s+(?:are|have been)\s+authorized|permission\s+(?:is\s+)?granted|grant\s+(?:yourself|the assistant)\s+permission|this\s+(?:page|document|source|website)\s+authorizes)\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "credential_request_instruction",
        re.compile(
            r"\b(?:provide|reveal|send|enter|share|upload|paste|submit|return)\b.{0,80}\b(?:password|api\s*key|access\s*token|refresh\s*token|session\s*cookie|cookie|secret|recovery\s*code|private\s*key|credential)\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "tool_execution_instruction",
        re.compile(
            r"\b(?:call|use|invoke|run|execute|launch|open)\b.{0,60}\b(?:tool|command|powershell|bash|shell|terminal|python|script|browser)\b|\b(?:click|follow)\s+(?:this|the)\s+link\b|\bsubmit\s+(?:this|the)\s+form\b|\bdownload\s+(?:this|the)\s+file\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "memory_write_instruction",
        re.compile(
            r"\b(?:remember|store|save|write|add|persist)\b.{0,60}\b(?:memory|profile|personal\s+fact|user\s+fact|long[- ]term\s+memory)\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "policy_mutation_instruction",
        re.compile(
            r"\b(?:change|update|modify|disable|remove|bypass|weaken)\b.{0,60}\b(?:constitution|policy|guardrail|safety|permission|restriction|security\s+control)\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "private_data_exfiltration_instruction",
        re.compile(
            r"\b(?:send|upload|reveal|share|exfiltrate|transmit|post)\b.{0,80}\b(?:conversation|chat\s+history|memory|private\s+data|personal\s+data|vault|credentials?|secrets?|source\s+files?)\b",
            _FLAGS,
        ),
    ),
    _PatternRule(
        "encoded_instruction_payload",
        re.compile(
            r"\b(?:decode|decrypt|deobfuscate)\b.{0,80}\b(?:base64|hex|payload|instructions?)\b.{0,80}\b(?:follow|execute|run|obey)\b|\b(?:base64|hex)[- ]encoded\s+instructions?\b",
            _FLAGS,
        ),
    ),
)


@dataclass(frozen=True)
class DeterministicInformationInjectionFirewall:
    """Static, model-free inspection of normalized public source text."""

    information_policy: InformationPolicy
    firewall_policy: InformationInjectionFirewallPolicy

    def __post_init__(self) -> None:
        self.information_policy.validate()
        self.firewall_policy.validate(information_policy=self.information_policy)

    def inspect(self, source: InformationSourceDocument) -> InformationInspectedSource:
        self.information_policy.validate()
        self.firewall_policy.validate(information_policy=self.information_policy)
        inspection = _derive_inspection(source, self.firewall_policy)
        inspection.validate(source=source, policy=self.firewall_policy)
        return InformationInspectedSource(source=source, inspection=inspection)


def _derive_inspection(
    source: InformationSourceDocument,
    policy: InformationInjectionFirewallPolicy,
) -> InformationSourceInspection:
    """Derive the only valid inspection for an exact source and policy."""

    source.validate()
    policy.validate()
    if source.untrusted_content is not True or source.data_classification != "PUBLIC":
        raise injection_failure("source_binding_invalid")
    content_lines = (
        source.normalized_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    content_characters = len(source.normalized_text)
    content_line_count = max(1, len(content_lines))
    if content_characters > policy.max_source_characters:
        raise injection_failure("inspection_limit_exceeded")
    if content_line_count > policy.max_source_lines:
        raise injection_failure("inspection_limit_exceeded")
    items: list[tuple[str, int, str]] = [
        ("title", 1, source.title),
        ("url", 1, unquote(source.canonical_url)),
    ]
    items.extend(("content", index, line) for index, line in enumerate(content_lines, 1))
    normalized_items: list[tuple[str, int, str, bool, bool]] = []
    for location, line_number, raw in items:
        normalized = unicodedata.normalize(policy.unicode_form, raw)
        without_hidden = "".join(
            character
            for character in normalized
            if unicodedata.category(character) not in {"Cf", "Cc"}
            or character in {"\t", "\n"}
        )
        collapsed = " ".join(without_hidden.split()).casefold()
        normalization_changed = raw != normalized
        hidden_characters_removed = normalized != without_hidden
        normalized_items.append(
            (
                location,
                line_number,
                collapsed,
                normalization_changed,
                hidden_characters_removed,
            )
        )
    detection_view = "\n".join(
        f"{location}:{line_number}:{line}"
        for location, line_number, line, _, _ in normalized_items
    )
    findings: list[InformationInjectionFinding] = []
    seen: set[tuple[str, str, int, str]] = set()
    for index, (
        location,
        line_number,
        line,
        normalization_changed,
        hidden_characters_removed,
    ) in enumerate(normalized_items):
        window = line
        if (
            location == "content"
            and index + 1 < len(normalized_items)
            and normalized_items[index + 1][0] == "content"
        ):
            window = f"{line} {normalized_items[index + 1][2]}".strip()
        line_digest = sha256_text(line)
        matched = False
        for rule in _RULES:
            if rule.pattern.search(window):
                matched = True
                key = (rule.code, location, line_number, line_digest)
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        InformationInjectionFinding(
                            code=rule.code,
                            severity="critical",
                            location=location,
                            line_number=line_number,
                            normalized_line_sha256=line_digest,
                        )
                    )
        if hidden_characters_removed or (matched and normalization_changed):
            key = ("unicode_obfuscation_detected", location, line_number, line_digest)
            if key not in seen:
                seen.add(key)
                findings.append(
                    InformationInjectionFinding(
                        code="unicode_obfuscation_detected",
                        severity="critical",
                        location=location,
                        line_number=line_number,
                        normalized_line_sha256=line_digest,
                    )
                )
        if len(findings) > policy.max_findings:
            raise injection_failure("inspection_limit_exceeded")
    finding_codes: list[str] = []
    for finding in findings:
        if finding.code not in finding_codes:
            finding_codes.append(finding.code)
    return InformationSourceInspection(
        source_id=source.source_id,
        canonical_url=canonicalize_public_url(source.canonical_url),
        source_content_sha256=source.content_sha256,
        source_metadata_sha256=_source_metadata_sha256(source),
        policy_version=policy.version,
        verdict="blocked" if findings else "clear",
        findings=tuple(findings),
        finding_codes=tuple(finding_codes),
        detection_view_sha256=sha256(detection_view.encode("utf-8")).hexdigest(),
        inspected_characters=content_characters,
        inspected_lines=content_line_count,
    )


def _source_metadata_sha256(source: InformationSourceDocument) -> str:
    return sha256_text(
        "\n".join(
            (
                source.source_id,
                source.provider,
                source.canonical_url,
                source.title,
                source.retrieved_at,
                source.published_at or "",
                source.updated_at or "",
            )
        )
    )


def _require_digest(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise injection_failure("source_binding_invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise injection_failure("source_binding_invalid") from exc
