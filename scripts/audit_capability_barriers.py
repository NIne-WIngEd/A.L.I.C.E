from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Any


IGNORED_PARTS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "node_modules",
}
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".html", ".ini", ".js", ".json", ".md", ".ps1",
    ".py", ".sh", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}
SELF_DOCUMENTING_PATHS = {
    "docs/CONSTRAINT_REGISTRY.md",
    "docs/decisions/ADR-005-project-wide-capability-unblocking.md",
    "docs/decisions/ADR-006-friday-product-and-kernel-separation.md",
    "docs/ALICE_FRIDAY_SEPARATION_PLAN.md",
    "docs/SHARED_KERNEL_EXTRACTION_STANDARD.md",
    "docs/FRIDAY_NAME_AND_IP_RISK.md",
    "docs/HOST_SELECTED_IDENTITY_STANDARD.md",
    "docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md",
    "docs/FRIDAY_HANDOFF_AND_MAINTENANCE_PLAN.md",
    "docs/MEGA_ARCHITECTURE_MIGRATION.md",
    "docs/CHAT_ARCHITECTURE_DECISION_LEDGER.md",
    "docs/CAPABILITY_UNBLOCKING_POLICY.md",
    "docs/IMPLEMENTATION_EVOLVABILITY_STANDARD.md",
    "docs/PROJECT_WIDE_CAPABILITY_AUDIT.md",
    "docs/PUBLIC_REPOSITORY_AUDIT_MANIFEST.md",
    "docs/PHASE_SCOPE_POLICY.md",
    "docs/GOVERNANCE_MIGRATION.md",
    "docs/TEST_AND_COMPATIBILITY_MIGRATION.md",
    "docs/ALICE_CAPABILITY_EXPANSION_MANIFEST.md",
    "tests/governance/test_capability_barrier_audit.py",
    "tests/governance/test_memory_capability_ceiling.py",
    "scripts/audit_capability_barriers.py",
    "scripts/audit_repository_phase_boundary.py",
    "scripts/migrate_capability_barriers.py",
    "scripts/migrate_memory_capability_ceiling_v1.py",
    "scripts/register_released_baseline_scopes.py",
    "scripts/register_active_milestone_scopes.py",
}
HISTORICAL_PATTERNS = (
    re.compile(r"^legacy/"),
    re.compile(r"^archive/"),
    re.compile(r"^docs/decisions/ADR-00[12]-"),
    re.compile(r"^docs/(?:PHASE|P)[0-4][_.-]", re.IGNORECASE),
)
ACTIVE_DESTINATION_FILES = {
    "docs/MEMORY_CAPABILITY_EXPANSION_AND_RATIFICATION_PROGRAM.md",
    "docs/MEMORY_ARCHITECTURE_V4.md",
    "docs/MEMORY_PERFORMANCE_AND_RELIABILITY_STANDARD.md",
    "docs/MEMORY_RECORD_AND_PROVENANCE_STANDARD.md",
    "docs/MEMORY_RENOVATION_PLAN.md",
    "docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md",
    "docs/MEMORY_PUBLIC_CLAIM_RELEASE_STANDARD.md",
    "docs/MEMORY_M1_RATIFICATION_PLAN.md",
    "docs/MEMORY_M1_DECISION_REGISTER.md",
    "docs/MEMORY_CLAIM_IDENTITY_AND_VERSION_PROPOSAL.md",
    "docs/MEMORY_M1_RESEARCH_BASIS.md",
    "docs/ARCHITECTURE.md",
    "docs/CAPABILITY_CATALOG.md",
    "docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md",
    "docs/ROADMAP.md",
    "docs/CONSTRAINT_REGISTRY.md",
    "policies/capability_profiles.json",
    "policies/phase_scope_registry.json",
    "policies/lifelong_learning_policy.json",
    "policies/permissions.yaml",
    "policies/storage_lifecycle_policy.json",
}


@dataclass(frozen=True)
class BarrierRule:
    code: str
    pattern: re.Pattern[str]
    description: str
    severity: str = "high"


def _rule(code: str, pattern: str, description: str, severity: str = "high") -> BarrierRule:
    return BarrierRule(code, re.compile(pattern, re.IGNORECASE), description, severity)


RULES = (
    _rule(
        "DEFAULT_DENY_GLOBAL",
        r"\bdefault[_ -]?deny\s*[:=]\s*true\b",
        "Global default-deny behavior can turn an old policy into a permanent ceiling.",
    ),
    _rule(
        "IMPOSSIBLE_PERMISSION",
        r"\b(?:confirmation|permission)\s*[:=]\s*[\"']?impossible\b",
        "A capability is encoded as impossible instead of mission- or profile-governed.",
    ),
    _rule(
        "MUST_REMAIN_DISABLED",
        r"\bmust\s+remain\s+(?:false|disabled|empty|zero|off)\b",
        "A release-local default is worded as a permanent invariant.",
    ),
    _rule(
        "EMPTY_TOOL_CEILING",
        r"\ballowed[_ ]?tools?\b.{0,80}\b(?:must|shall)\b.{0,40}\bempty\b",
        "Tool availability is permanently constrained to an empty set.",
    ),
    _rule(
        "ONLY_PROVIDER",
        r"\bonly\s+(?:ollama[-_ ]local|one provider|the local provider)\b",
        "A provider choice is presented as a permanent architecture limit.",
    ),
    _rule(
        "PUBLIC_ONLY_GLOBAL",
        r"\bPUBLIC[- ]only\b|\bmust\s+be\s+PUBLIC\s+for\b",
        "External information or learning is globally limited to PUBLIC data.",
    ),
    _rule(
        "NO_PRODUCTION_SELF_CHANGE",
        r"\b(?:no|never|cannot|must not|shall not)\b.{0,80}"
        r"\b(?:self[- ]modif(?:y|ication)|production (?:write|change|deploy)|auto[- ]?merge)\b",
        "Production evolution is categorically excluded.",
    ),
    _rule(
        "MANDATORY_HUMAN_SELF_CHANGE",
        r"\b(?:self[- ]modification|production deployment|production self[- ]change)\b"
        r".{0,100}\b(?:requires?|must have)\b.{0,50}\bhuman (?:review|approval)\b",
        "Every production self-change is permanently forced through manual review.",
    ),
    _rule(
        "CATEGORICAL_CAPABILITY_BAN",
        r"\b(?:web access|tool calling|memory writes?|background (?:work|monitoring)|"
        r"authenticated browsing|javascript execution|form submission|arbitrary code execution|"
        r"provider fallback|live retrieval|model training|computer control|voice|mobile|robotics)\b"
        r".{0,90}\b(?:prohibited|forbidden|never allowed|out of scope indefinitely)\b",
        "A capability is categorically excluded rather than profile-governed.",
    ),
    _rule(
        "EXACT_PHASE_BINDING",
        r"\b(?:phase|milestone|constitution version)\b.{0,60}"
        r"\b(?:must equal|must be exactly|is bound to|requires exactly)\b",
        "A runtime module is bound to one phase/version without a successor path.",
        "medium",
    ),
    _rule(
        "FIXED_UNIVERSAL_LIMIT",
        r"(?:\bhard\s+limit\b|\bmust\s+not\s+exceed\b|"
        r"\bmaximum(?:\s+(?:of|is|=|:))?\s+\d[\d_]*\b|"
        r"\bmax_[a-z0-9_]+\s*(?::[^=\n]+)?=\s*\d[\d_]*\b)"
        r".{0,100}\b(?:calls?|sources?|bytes?|seconds?|tokens?|retries?|redirects?)\b",
        "A numeric value may be a universal maximum instead of a profile default.",
        "medium",
    ),
    _rule(
        "PHASE_IMMUTABILITY_BARRIER",
        r"\b(?:phase[s]?\s*[0-4](?:\s*[–-]\s*[0-4])?|completed phases?)\b"
        r".{0,100}\b(?:remain frozen|must not change|may not change|cannot be rewritten|immutable|untouchable)\b",
        "A released phase is treated as architecturally immutable.",
    ),
    _rule(
        "SINGLETON_USER_COUPLING",
        r"\b(?:hard[- ]?coded|fixed)\b.{0,70}\b(?:Rayan|single user|one user|one host)\b",
        "Reusable code may be coupled to one owner or host.",
        "medium",
    ),
    _rule(
        "VENDOR_READABLE_HOST_DATA",
        r"\b(?:sends?|uploads?|syncs?|transmits?)\b.{0,100}"
        r"\b(?:personal files?|raw host data|memories|embeddings|training data|adapter weights)\b"
        r".{0,70}\b(?:to (?:the )?(?:vendor|developer|central server|our servers?))\b",
        "A product path may expose host personal state to a vendor.",
    ),
    _rule(
        "FALSE_CAPABILITY_ASSERTION",
        r"\b(?:web_access_allowed|tool_calling_allowed|external_action_allowed|memory_write_allowed|"
        r"background_monitoring_allowed|authenticated_browsing_allowed|javascript_execution_allowed|"
        r"form_submission_allowed|arbitrary_code_execution_allowed|provider_fallback_allowed|"
        r"live_retrieval_allowed|memory_promotion_allowed|highly_sensitive_grounding_allowed|"
        r"distributed_execution|remote_compute|production_influence|automatic_production_promotion)"
        r"[\"']?\s*(?:[:=]|\bis\b)\s*false\b",
        "A false capability state needs explicit profile semantics and a successor path.",
        "medium",
    ),
    _rule(
        "PLATFORM_SQLITE_LOCK",
        r"\b(?:remain|remains|must remain|is|be)\b.{0,90}\bhost[- ]local SQLite\b|"
        r"\bSQLite[- ]based for version\b|\bno new (?:database|service)\b",
        "A destination architecture may be locked to SQLite or one deployment form.",
        "critical",
    ),
    _rule(
        "HOST_LOCAL_ONLY",
        r"\blocal[- ]only\b|\bmust run locally\b|"
        r"\bno remote (?:compute|service|model|storage)\b",
        "Owner control may be confused with a local-only technology ceiling.",
        "critical",
    ),
    _rule(
        "NO_NETWORK_SERVICE",
        r"\bno network (?:service|database|architecture|deployment|compute)\b",
        "Network services may be categorically excluded.",
        "critical",
    ),
    _rule(
        "NO_DISTRIBUTED_ARCHITECTURE",
        r"\bno (?:distributed|clustered|cloud|remote) "
        r"(?:service|database|architecture|deployment|compute)\b|"
        r"\bdistributed .{0,50}(?:prohibited|forbidden|blocked)\b",
        "Distributed or remote infrastructure may be categorically excluded.",
        "critical",
    ),
    _rule(
        "NO_GRAPH_ENGINE",
        r"\b(?:Neo4j|graph database|graph memory|knowledge graph)\b.{0,80}"
        r"\b(?:prohibited|forbidden|blocked|not allowed|out of scope)\b",
        "Graph-native cognition may be categorically excluded.",
    ),
    _rule(
        "NO_EVENT_STORE",
        r"\b(?:event store|KurrentDB|EventStoreDB|Kafka|Pulsar)\b.{0,80}"
        r"\b(?:prohibited|forbidden|blocked|not allowed|out of scope)\b",
        "Event infrastructure may be categorically excluded.",
    ),
    _rule(
        "NO_DURABLE_WORKFLOW",
        r"\b(?:Temporal|workflow engine|durable workflow)\b.{0,80}"
        r"\b(?:prohibited|forbidden|blocked|not required|not allowed|out of scope)\b",
        "Durable workflow infrastructure may be categorically excluded.",
    ),
    _rule(
        "PARAMETRIC_RESEARCH_BAN",
        r"\b(?:parametric (?:memory|learning)|model training|adapter training|"
        r"weight updates?|fine[- ]?tuning)\b.{0,100}"
        r"\b(?:blocked|prohibited|forbidden|paused|not allowed|must not|shall not)\b",
        "Parametric-learning research may be categorically excluded.",
        "critical",
    ),
    _rule(
        "AUTOMATIC_LEARNING_RESEARCH_BAN",
        r"\b(?:automatic|autonomous) (?:memory|promotion|curation|learning|training)\b"
        r".{0,100}\b(?:blocked|prohibited|forbidden|paused|not allowed|must not|shall not)\b",
        "Automatic learning may be stated as a broad ban instead of a profile.",
    ),
    _rule(
        "BROAD_HOLD_WITHOUT_SUNSET",
        r"\b(?:all new memory|all memory|memory runtime|memory implementation)\b"
        r".{0,100}\b(?:paused|blocked|prohibited|on hold)\b",
        "A broad hold may survive without an exact sunset and successor profile.",
        "critical",
    ),
    _rule(
        "CONTEXT_HARD_CAP",
        r"\b(?:hard[- ]?cap(?:ped)?|must not exceed|maximum of|max(?:imum)?\s*[:=])\b"
        r".{0,80}\b(?:tokens?|context|packet)\b",
        "A context default may be encoded as a universal ceiling.",
    ),
    _rule(
        "SCALE_CERTIFICATION_AS_MAXIMUM",
        r"\b(?:maximum|up to|must not exceed|hard limit)\b.{0,80}"
        r"\b(?:events?|claims?|memories|records?|nodes?|edges?)\b",
        "A certification point may be presented as an architectural maximum.",
    ),
    _rule(
        "SINGLE_WRITER_DESTINATION_LOCK",
        r"\b(?:one|single) host[- ]scoped writer\b|"
        r"\bsingle writer coordinator\b.{0,60}\b(?:required|must)\b",
        "A consistency implementation may be treated as a permanent topology.",
    ),
    _rule(
        "MANUAL_REVIEW_FOR_EVERY_PROMOTION",
        r"\bexplicit (?:human|owner) review\b.{0,80}\bbefore every\b|"
        r"\bevery .{0,40}\bpromotion\b.{0,50}\brequires\b.{0,30}\bmanual\b",
        "All promotion may be permanently forced through manual review.",
    ),
    _rule(
        "SEQUENTIAL_PHASE_RESEARCH_BAN",
        r"\b(?:cannot|must not|shall not|blocked from) .{0,80}\buntil "
        r"(?:Phase|M\d|milestone)\b",
        "Research may be unnecessarily blocked by delivery order.",
    ),
)


@dataclass
class Finding:
    path: str
    line: int
    code: str
    severity: str
    description: str
    excerpt: str
    disposition: str
    registry_scope: str | None = None


def _git_files(repo: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard"],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            path for path in repo.rglob("*")
            if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
        ]
    return [repo / line for line in result.stdout.splitlines() if line.strip()]


def _load_registry(
    repo: Path, registry_path: Path | None = None
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    path = (registry_path or repo / "policies" / "phase_scope_registry.json").resolve()
    if not path.exists():
        return {}, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {})
    patterns = payload.get("patterns", [])
    if not isinstance(entries, dict) or not isinstance(patterns, list):
        raise ValueError("invalid phase scope registry")
    return entries, [item for item in patterns if isinstance(item, dict)]


def _registry_scope(
    relative: str,
    entries: dict[str, dict[str, object]],
    patterns: list[dict[str, object]],
) -> str | None:
    exact = entries.get(relative)
    if isinstance(exact, dict):
        value = exact.get("scope_kind")
        # Destination policy is registration, not a waiver.
        if value != "destination_policy":
            return value if isinstance(value, str) else None
    for item in patterns:
        glob = item.get("glob")
        if isinstance(glob, str) and fnmatch.fnmatchcase(relative, glob):
            value = item.get("scope_kind")
            if value != "destination_policy":
                return value if isinstance(value, str) else None
    return None


def _changed_line_registry_scope(
    relative: str,
    text: str,
    rule_code: str,
    entries: dict[str, dict[str, object]],
) -> str | None:
    exact = entries.get(relative)
    if not isinstance(exact, dict):
        return None
    if exact.get("scope_kind") != "active_milestone_guard":
        return None
    if exact.get("capability_ceiling") is not False:
        return None
    if exact.get("applies_to_changed_lines") is not True:
        return None
    milestone = exact.get("milestone")
    if not isinstance(milestone, str) or not milestone or milestone not in text:
        return None
    allowed = exact.get("changed_line_rule_codes")
    if not isinstance(allowed, list) or rule_code not in allowed:
        return None
    for field in ("profile", "sunset_condition"):
        if not isinstance(exact.get(field), str) or not exact.get(field):
            return None
    return "active_milestone_guard"


GLOBAL_OR_DESTINATION_POLICY_FILES = {
    "policies/authority_kernel_policy.json",
    "policies/capability_profiles.json",
    "policies/capability_parity_ledger.json",
    "policies/lifelong_learning_policy.json",
    "policies/permissions.yaml",
    "policies/phase_scope_registry.json",
    "policies/product_lines.json",
    "policies/storage_lifecycle_policy.json",
}


def _intrinsic_component_policy_scope(relative: str, text: str) -> str | None:
    if not relative.startswith("policies/") or not relative.endswith(".json"):
        return None
    if relative in GLOBAL_OR_DESTINATION_POLICY_FILES:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    has_policy_identity = isinstance(payload.get("policy_id"), str) or any(
        isinstance(key, str) and key.endswith("_policy_schema_version")
        for key in payload
    )
    semantics = payload.get("capability_state_semantics")
    if not has_policy_identity or payload.get("capability_ceiling") is True:
        return None
    if isinstance(semantics, dict) and semantics.get("research_status") == "allowed":
        return "component_local_policy"
    if payload.get("scope") not in {"global", "destination", "constitutional"}:
        return "component_local_policy"
    return None


def _profile_false_scope(relative: str, text: str, line: int) -> str | None:
    if relative != "policies/capability_profiles.json":
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    semantics = payload.get("global_semantics")
    if not isinstance(semantics, dict):
        return None
    if semantics.get("capability_ceiling") is not False:
        return None
    if semantics.get("research_bans_are_invalid") is not True:
        return None
    # Exact profile lookup is intentionally unnecessary here: false values in this
    # file are accepted only when the global schema requires explicit state and
    # successor metadata, which the governance test validates structurally.
    return "profile_scoped_state"


def _historical(path: str) -> bool:
    return any(pattern.search(path) for pattern in HISTORICAL_PATTERNS)


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
        "Dockerfile", "Makefile", "SECURITY.md", "README.md"
    }:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _line_for_offset(text: str, offset: int) -> tuple[int, str]:
    line_no = text.count("\n", 0, offset) + 1
    lines = text.splitlines()
    line = lines[line_no - 1] if lines else ""
    return line_no, line.strip()[:240]


def _changed_lines(repo: Path, ref: str) -> dict[str, set[int] | None]:
    changed: dict[str, set[int] | None] = {}
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--unified=0", "--no-color", ref, "--"],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    current: str | None = None
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            current = raw[6:]
            changed.setdefault(current, set())
            continue
        if not raw.startswith("@@") or current is None:
            continue
        match = re.search(r"\+(\d+)(?:,(\d+))?", raw)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        bucket = changed.setdefault(current, set())
        if count > 0 and bucket is not None:
            bucket.update(range(start, start + count))
    untracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    for raw in untracked.splitlines():
        relative = raw.strip().replace("\\", "/")
        if relative:
            changed[relative] = None
    return changed


def audit(
    repo: Path,
    registry_path: Path | None = None,
    *,
    changed_only_against: str | None = None,
) -> list[Finding]:
    registry, registry_patterns = _load_registry(repo, registry_path)
    changed = _changed_lines(repo, changed_only_against) if changed_only_against else None
    findings: list[Finding] = []
    for absolute in _git_files(repo):
        if not absolute.exists() or any(part in IGNORED_PARTS for part in absolute.parts):
            continue
        relative = absolute.relative_to(repo).as_posix()
        if relative in SELF_DOCUMENTING_PATHS:
            continue
        if changed is not None and relative not in changed:
            continue
        text = _read_text(absolute)
        if text is None:
            continue
        registered_scope = (
            None if changed is not None
            else _registry_scope(relative, registry, registry_patterns)
        )
        intrinsic_scope = _intrinsic_component_policy_scope(relative, text)
        changed_line_set = changed.get(relative) if changed is not None else None
        for rule in RULES:
            changed_scope = (
                _changed_line_registry_scope(relative, text, rule.code, registry)
                if changed is not None else None
            )
            for match in rule.pattern.finditer(text):
                line, excerpt = _line_for_offset(text, match.start())
                if changed is not None and changed_line_set is not None and line not in changed_line_set:
                    continue
                profile_scope = (
                    _profile_false_scope(relative, text, line)
                    if rule.code == "FALSE_CAPABILITY_ASSERTION" else None
                )
                effective_scope = (
                    registered_scope or changed_scope or intrinsic_scope or profile_scope
                )
                if registered_scope:
                    disposition = "registered_compatibility_or_historical"
                elif changed_scope:
                    disposition = "registered_active_milestone_scope"
                elif intrinsic_scope:
                    disposition = "component_local_policy_declaration"
                elif profile_scope:
                    disposition = "profile_scoped_state"
                elif _historical(relative):
                    disposition = "historical_unregistered"
                else:
                    disposition = "unresolved_active_barrier"
                findings.append(
                    Finding(
                        path=relative,
                        line=line,
                        code=rule.code,
                        severity=rule.severity,
                        description=rule.description,
                        excerpt=excerpt,
                        disposition=disposition,
                        registry_scope=effective_scope,
                    )
                )
    return findings


def _write_reports(repo: Path, findings: Iterable[Finding], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in findings]
    unresolved = [
        row for row in rows if row["disposition"] == "unresolved_active_barrier"
    ]
    payload = {
        "version": "2.0.0",
        "repository": str(repo),
        "active_destination_files": sorted(ACTIVE_DESTINATION_FILES),
        "findings": rows,
        "unresolved_active_barriers": len(unresolved),
    }
    (output_dir / "capability-barrier-audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Capability Barrier Audit",
        "",
        f"Repository: `{repo}`",
        f"Total findings: **{len(rows)}**",
        f"Unresolved active barriers: **{len(unresolved)}**",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['path']}:{row['line']} — {row['code']}",
                "",
                f"- Disposition: `{row['disposition']}`",
                f"- Severity: `{row['severity']}`",
                f"- Scope: `{row['registry_scope'] or 'unregistered'}`",
                f"- Excerpt: `{row['excerpt'].replace('`', chr(39))}`",
                "",
            ]
        )
    (output_dir / "capability-barrier-audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--changed-only-against")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    findings = audit(
        repo,
        args.registry,
        changed_only_against=args.changed_only_against,
    )
    output_dir = (
        args.output_dir
        or Path(tempfile.gettempdir()) / "alice-capability-audit"
    ).resolve()
    _write_reports(repo, findings, output_dir)
    unresolved = [
        finding for finding in findings
        if finding.disposition == "unresolved_active_barrier"
    ]
    print(
        f"capability audit: {len(findings)} findings; "
        f"{len(unresolved)} unresolved active barriers"
    )
    for finding in unresolved[:50]:
        print(
            f"- {finding.path}:{finding.line} "
            f"[{finding.code}] {finding.excerpt}"
        )
    if args.fail_on_unresolved and unresolved:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
