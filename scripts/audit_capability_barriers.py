from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SELF_DOCUMENTING_PATHS = {
    "docs/CAPABILITY_UNBLOCKING_POLICY.md",
    "docs/IMPLEMENTATION_EVOLVABILITY_STANDARD.md",
    "docs/PROJECT_WIDE_CAPABILITY_AUDIT.md",
    "docs/PUBLIC_REPOSITORY_AUDIT_MANIFEST.md",
    "docs/PHASE_SCOPE_POLICY.md",
    "docs/CONSTRAINT_REGISTRY.md",
    "docs/GOVERNANCE_MIGRATION.md",
    "docs/TEST_AND_COMPATIBILITY_MIGRATION.md",
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
    "tests/governance/test_capability_barrier_audit.py",
    "policies/capability_profiles.json",
    "policies/phase_scope_registry.json",
    "scripts/audit_capability_barriers.py",
    "scripts/migrate_capability_barriers.py",
    "scripts/register_released_baseline_scopes.py",
    "scripts/register_active_milestone_scopes.py",
}
HISTORICAL_PATTERNS = (
    re.compile(r"^legacy/"),
    re.compile(r"^archive/"),
    re.compile(r"^docs/decisions/ADR-00[12]-"),
    re.compile(r"^docs/(?:PHASE|P)[0-4][_.-]", re.IGNORECASE),
)


@dataclass(frozen=True)
class BarrierRule:
    code: str
    pattern: re.Pattern[str]
    description: str
    severity: str = "high"


RULES = (
    BarrierRule(
        "DEFAULT_DENY_GLOBAL",
        re.compile(r"\bdefault[_ -]?deny\s*[:=]\s*true\b", re.IGNORECASE),
        "Global default-deny behavior can turn an old phase policy into a permanent ceiling.",
    ),
    BarrierRule(
        "IMPOSSIBLE_PERMISSION",
        re.compile(r"\b(?:confirmation|permission)\s*[:=]\s*[\"']?impossible\b", re.IGNORECASE),
        "A capability is encoded as impossible instead of mission- or profile-governed.",
    ),
    BarrierRule(
        "MUST_REMAIN_DISABLED",
        re.compile(r"\bmust\s+remain\s+(?:false|disabled|empty|zero|off)\b", re.IGNORECASE),
        "A release-local default is worded as a permanent invariant.",
    ),
    BarrierRule(
        "EMPTY_TOOL_CEILING",
        re.compile(r"\ballowed[_ ]?tools?\b.{0,80}\b(?:must|shall)\b.{0,40}\bempty\b", re.IGNORECASE),
        "Tool availability is permanently constrained to an empty set.",
    ),
    BarrierRule(
        "ONLY_PROVIDER",
        re.compile(r"\bonly\s+(?:ollama[-_ ]local|one provider|the local provider)\b", re.IGNORECASE),
        "A provider choice is presented as a permanent architecture limit.",
    ),
    BarrierRule(
        "PUBLIC_ONLY_GLOBAL",
        re.compile(r"\bPUBLIC[- ]only\b|\bmust\s+be\s+PUBLIC\s+for\b", re.IGNORECASE),
        "External information or learning is globally limited to PUBLIC data.",
    ),
    BarrierRule(
        "NO_PRODUCTION_SELF_CHANGE",
        re.compile(
            r"\b(?:no|never|cannot|must not|shall not)\b.{0,80}"
            r"\b(?:self[- ]modif(?:y|ication)|production (?:write|change|deploy)|auto[- ]?merge)\b",
            re.IGNORECASE,
        ),
        "Production evolution or self-modification is categorically prohibited.",
    ),
    BarrierRule(
        "MANDATORY_HUMAN_SELF_CHANGE",
        re.compile(
            r"\b(?:self[- ]modification|production deployment|production self[- ]change)\b"
            r".{0,100}\b(?:requires?|must have)\b.{0,50}\bhuman (?:review|approval)\b",
            re.IGNORECASE,
        ),
        "Every production self-change is permanently forced through manual review.",
    ),
    BarrierRule(
        "CATEGORICAL_CAPABILITY_BAN",
        re.compile(
            r"\b(?:web access|tool calling|memory writes?|background (?:work|monitoring)|"
            r"authenticated browsing|javascript execution|form submission|arbitrary code execution|"
            r"provider fallback|live retrieval|model training|computer control|voice|mobile|robotics)\b"
            r".{0,90}\b(?:prohibited|forbidden|never allowed|out of scope indefinitely)\b",
            re.IGNORECASE,
        ),
        "A capability is categorically banned rather than scoped to a release profile.",
    ),
    BarrierRule(
        "EXACT_PHASE_BINDING",
        re.compile(
            r"\b(?:phase|milestone|constitution version)\b.{0,60}"
            r"\b(?:must equal|must be exactly|is bound to|requires exactly)\b",
            re.IGNORECASE,
        ),
        "A runtime module is bound to one phase/version with no successor profile path.",
        "medium",
    ),
    BarrierRule(
        "FIXED_UNIVERSAL_LIMIT",
        re.compile(
            r"(?:\bhard\s+limit\b|\bmust\s+not\s+exceed\b|"
            r"\bmaximum(?:\s+(?:of|is|=|:))?\s+\d[\d_]*\b|"
            r"\bmax_[a-z0-9_]+\s*(?::[^=\n]+)?=\s*\d[\d_]*\b)"
            r".{0,100}\b(?:calls?|sources?|bytes?|seconds?|tokens?|retries?|redirects?)\b",
            re.IGNORECASE,
        ),
        "A numeric limit may be acting as a universal maximum instead of a profile default.",
        "medium",
    ),
    BarrierRule(
        "PHASE_IMMUTABILITY_BARRIER",
        re.compile(
            r"\b(?:phase[s]?\s*[0-4](?:\s*[–-]\s*[0-4])?|completed phases?)\b"
            r".{0,100}\b(?:remain frozen|must not change|may not change|cannot be rewritten|immutable|untouchable)\b",
            re.IGNORECASE,
        ),
        "An earlier phase is treated as architecturally immutable rather than a migratable released baseline.",
    ),
    BarrierRule(
        "SINGLETON_USER_COUPLING",
        re.compile(
            r"\b(?:hard[- ]?coded|fixed)\b.{0,70}\b(?:Rayan|single user|one user|one host)\b",
            re.IGNORECASE,
        ),
        "Reusable code may be coupled to one owner or host instead of explicit product and host identity.",
        "medium",
    ),
    BarrierRule(
        "VENDOR_READABLE_HOST_DATA",
        re.compile(
            r"\b(?:sends?|uploads?|syncs?|transmits?)\b.{0,100}"
            r"\b(?:personal files?|raw host data|memories|embeddings|training data|adapter weights)\b"
            r".{0,70}\b(?:to (?:the )?(?:vendor|developer|central server|our servers?))\b",
            re.IGNORECASE,
        ),
        "A product path may make host personal state readable by the vendor without a scoped optional mode.",
    ),
    BarrierRule(
        "FALSE_CAPABILITY_ASSERTION",
        re.compile(
            r"\b(?:web_access_allowed|tool_calling_allowed|external_action_allowed|memory_write_allowed|"
            r"background_monitoring_allowed|authenticated_browsing_allowed|javascript_execution_allowed|"
            r"form_submission_allowed|arbitrary_code_execution_allowed|provider_fallback_allowed|"
            r"live_retrieval_allowed|memory_promotion_allowed|highly_sensitive_grounding_allowed)"
            r"[\"']?\s*(?:[:=]|\bis\b)\s*false\b",
            re.IGNORECASE,
        ),
        "A capability is hard-coded false; it must be profile-scoped or registered as compatibility behavior.",
        "medium",
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
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            path for path in repo.rglob("*")
            if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
        ]
    return [repo / line for line in result.stdout.splitlines() if line.strip()]


def _load_registry(repo: Path, registry_path: Path | None = None) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    path = (registry_path or (repo / "policies" / "phase_scope_registry.json")).resolve()
    if not path.exists():
        return {}, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {})
    patterns = payload.get("patterns", [])
    if not isinstance(entries, dict):
        raise ValueError("phase scope registry entries must be an object")
    if not isinstance(patterns, list) or any(not isinstance(item, dict) for item in patterns):
        raise ValueError("phase scope registry patterns must be a list of objects")
    return entries, list(patterns)


def _registry_scope(
    relative: str,
    entries: dict[str, dict[str, object]],
    patterns: list[dict[str, object]],
) -> str | None:
    exact = entries.get(relative)
    if isinstance(exact, dict):
        value = exact.get("scope_kind")
        return value if isinstance(value, str) else None
    for item in patterns:
        glob = item.get("glob")
        if isinstance(glob, str) and fnmatch.fnmatchcase(relative, glob):
            value = item.get("scope_kind")
            return value if isinstance(value, str) else None
    return None


def _changed_line_registry_scope(
    relative: str,
    text: str,
    rule_code: str,
    entries: dict[str, dict[str, object]],
) -> str | None:
    """Allow only exact, reviewable active-milestone guards on changed lines.

    Pattern entries and ordinary compatibility registrations never waive changed
    lines.  A changed-line scope must be an exact file entry, explicitly opt in,
    name the active milestone in the source, and list the exact scanner rule codes
    it is permitted to scope.
    """
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
    if not isinstance(exact.get("profile"), str) or not exact.get("profile"):
        return None
    if not isinstance(exact.get("sunset_condition"), str) or not exact.get("sunset_condition"):
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
}


def _intrinsic_component_policy_scope(relative: str, text: str) -> str | None:
    """Recognize bounded component-policy declarations without waiving global policy."""
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
    if not has_policy_identity:
        return None
    if payload.get("capability_ceiling") is True or payload.get("scope") in {
        "global", "destination", "constitutional"
    }:
        return None
    return "component_local_policy"


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
    """Return changed new-file line numbers; None means every line of an untracked file."""
    changed: dict[str, set[int] | None] = {}
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--unified=0", "--no-color", ref, "--"],
        check=True,
        capture_output=True,
        text=True,
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
        if count <= 0:
            continue
        bucket = changed.setdefault(current, set())
        if bucket is not None:
            bucket.update(range(start, start + count))

    untracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
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
        # In changed-only mode, ordinary compatibility entries and patterns do not
        # waive new lines.  Only an exact active_milestone_guard entry may scope
        # specifically listed rule codes after milestone and sunset validation.
        registered_scope = None if changed is not None else _registry_scope(relative, registry, registry_patterns)
        intrinsic_scope = _intrinsic_component_policy_scope(relative, text)
        changed_line_set = changed.get(relative) if changed is not None else None
        for rule in RULES:
            changed_scope = (
                _changed_line_registry_scope(relative, text, rule.code, registry)
                if changed is not None
                else None
            )
            for match in rule.pattern.finditer(text):
                line, excerpt = _line_for_offset(text, match.start())
                if changed is not None and changed_line_set is not None and line not in changed_line_set:
                    continue
                effective_scope = registered_scope or changed_scope or intrinsic_scope
                if registered_scope:
                    disposition = "registered_compatibility_or_historical"
                elif changed_scope:
                    disposition = "registered_active_milestone_scope"
                elif intrinsic_scope:
                    disposition = "component_local_policy_declaration"
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
    (output_dir / "capability-barrier-audit.json").write_text(
        json.dumps({"version": "1.3.0", "findings": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    unresolved = [row for row in rows if row["disposition"] == "unresolved_active_barrier"]
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
                f"- Excerpt: `{row['excerpt'].replace('`', "'")}`",
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
    parser.add_argument("--registry", type=Path, help="Use an explicit phase-scope registry, including when auditing another worktree.")
    parser.add_argument(
        "--changed-only-against",
        help="Audit only added/modified lines relative to this Git ref; registry entries do not waive changed lines.",
    )
    parser.add_argument("--fail-on-unresolved", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    findings = audit(
        repo,
        args.registry,
        changed_only_against=args.changed_only_against,
    )
    output_dir = (args.output_dir or Path(tempfile.gettempdir()) / "alice-capability-audit").resolve()
    _write_reports(repo, findings, output_dir)
    unresolved = [f for f in findings if f.disposition == "unresolved_active_barrier"]
    print(
        f"capability audit: {len(findings)} findings; "
        f"{len(unresolved)} unresolved active barriers"
    )
    for finding in unresolved[:50]:
        print(f"- {finding.path}:{finding.line} [{finding.code}] {finding.excerpt}")
    if args.fail_on_unresolved and unresolved:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
