from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


TEXT_SUFFIXES = {
    ".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".csv", ".tsv", ".html", ".css", ".js", ".ts", ".tsx", ".sh",
    ".ps1", ".bat", ".lmp",
}
FORBIDDEN_PATH_SUFFIXES = {
    ".sqlite", ".sqlite3", ".db", ".pem", ".p12", ".pfx", ".key", ".pyc",
}
FORBIDDEN_PATH_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea",
}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_pat": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
CONFLICT_PATTERN = re.compile(r"^(?:<<<<<<<|=======|>>>>>>>)", re.MULTILINE)
DESTINATION_LOCK_RULES = {
    "PLATFORM_SQLITE_LOCK": re.compile(
        r"\b(?:remain|remains|must remain|is|be)\b.{0,90}\bhost[- ]local SQLite\b|"
        r"\bSQLite[- ]based for version\b|\bno new (?:database|service)\b",
        re.IGNORECASE,
    ),
    "HOST_LOCAL_ONLY": re.compile(
        r"\blocal[- ]only\b|\bmust run locally\b|"
        r"\bno remote (?:compute|service|model|storage)\b",
        re.IGNORECASE,
    ),
    "DISTRIBUTED_EXCLUSION": re.compile(
        r"\bno (?:network|distributed|clustered|cloud|remote) "
        r"(?:service|database|architecture|deployment|compute)\b",
        re.IGNORECASE,
    ),
    "PARAMETRIC_RESEARCH_BAN": re.compile(
        r"\b(?:parametric (?:memory|learning)|model training|adapter training|"
        r"weight updates?|fine[- ]?tuning)\b.{0,100}"
        r"\b(?:blocked|prohibited|forbidden|paused|not allowed|must not|shall not)\b",
        re.IGNORECASE,
    ),
    "CONTEXT_HARD_CAP": re.compile(
        r"\b(?:hard[- ]?cap(?:ped)?|must not exceed|maximum of|max(?:imum)?\s*[:=])\b"
        r".{0,80}\b(?:tokens?|context|packet)\b",
        re.IGNORECASE,
    ),
}
ACTIVE_DESTINATION_DOCS = (
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
)
CAPABILITY_MARKERS = {
    "docs/ALICE_CAPABILITY_EXPANSION_MANIFEST.md":
        "owner-sovereign, local-capable, deployment-unbounded",
    "docs/MEMORY_CAPABILITY_EXPANSION_AND_RATIFICATION_PROGRAM.md":
        "Capability gates are activation criteria",
    "docs/MEMORY_ARCHITECTURE_V4.md":
        "Capability-First Polyglot Cognitive Fabric",
    "docs/MEMORY_PERFORMANCE_AND_RELIABILITY_STANDARD.md":
        "They are not an architectural maximum",
    "docs/MEMORY_RECORD_AND_PROVENANCE_STANDARD.md":
        "Store and Capability Fabric Registration",
    "docs/MEMORY_RENOVATION_PLAN.md":
        "parallel research and independently activated production profiles",
    "docs/MEMORY_PUBLIC_CLAIM_RELEASE_STANDARD.md":
        "production_profile_enabled",
    "docs/MEMORY_M1_RATIFICATION_PLAN.md":
        "M1-DX0",
    "docs/MEMORY_M1_DECISION_REGISTER.md":
        "M1-DX0 through M1-D9 owner-ratified",
    "docs/CONSTRAINT_REGISTRY.md":
        "capability_ceiling: false",
}


def git_paths(repo: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8", errors="strict")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".toml":
        return "toml"
    if suffix in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}:
        return "text"
    return "binary_or_unknown"


def add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    path: str,
    code: str,
    detail: str,
) -> None:
    findings.append(
        {"severity": severity, "path": path, "code": code, "detail": detail}
    )


def inspect_text(
    *,
    rel: str,
    text: str,
    category: str,
    findings: list[dict[str, Any]],
) -> str:
    parse_status = "not_applicable"
    if CONFLICT_PATTERN.search(text):
        add_finding(
            findings, "critical", rel,
            "MERGE_CONFLICT_MARKER", "merge conflict marker found",
        )
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            add_finding(
                findings, "critical", rel,
                f"SECRET_{name.upper()}", "high-confidence secret pattern found",
            )
    trailing = sum(1 for line in text.splitlines() if line.endswith((" ", "\t")))
    if trailing:
        add_finding(
            findings, "advisory", rel,
            "TRAILING_WHITESPACE", f"{trailing} line(s)",
        )
    try:
        if category == "python":
            ast.parse(text, filename=rel)
            parse_status = "passed"
        elif category == "json":
            json.loads(text)
            parse_status = "passed"
        elif category == "yaml":
            if yaml is None:
                add_finding(
                    findings, "major", rel,
                    "YAML_PARSER_UNAVAILABLE", "PyYAML is required",
                )
                parse_status = "not_run"
            else:
                yaml.safe_load(text)
                parse_status = "passed"
        elif category == "toml":
            tomllib.loads(text)
            parse_status = "passed"
        else:
            parse_status = "readable"
    except Exception as exc:  # noqa: BLE001
        add_finding(
            findings, "critical", rel,
            "PARSE_FAILURE", f"{type(exc).__name__}: {exc}",
        )
        parse_status = "failed"
    return parse_status


def require_marker(
    *,
    repo: Path,
    rel: str,
    marker: str,
    findings: list[dict[str, Any]],
) -> None:
    path = repo / rel
    if not path.is_file():
        add_finding(
            findings, "critical", rel,
            "REQUIRED_FILE_MISSING", "required controlling file missing",
        )
        return
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        add_finding(
            findings, "critical", rel,
            "REQUIRED_MARKER_MISSING", marker,
        )


def inspect_destination_policy(
    repo: Path,
    findings: list[dict[str, Any]],
) -> None:
    for rel in ACTIVE_DESTINATION_DOCS:
        path = repo / rel
        if not path.is_file():
            add_finding(
                findings, "critical", rel,
                "ACTIVE_DESTINATION_FILE_MISSING",
                "capability-first remediation matrix file is missing",
            )
            continue
        text = path.read_text(encoding="utf-8")
        for code, pattern in DESTINATION_LOCK_RULES.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                excerpt = text.splitlines()[line - 1].strip()[:220]
                add_finding(
                    findings, "critical", rel,
                    code, f"line {line}: {excerpt}",
                )

    for rel, marker in CAPABILITY_MARKERS.items():
        require_marker(
            repo=repo, rel=rel, marker=marker, findings=findings
        )

    profiles_path = repo / "policies" / "capability_profiles.json"
    if profiles_path.is_file():
        try:
            profiles_payload = json.loads(
                profiles_path.read_text(encoding="utf-8")
            )
            semantics = profiles_payload["global_semantics"]
            if semantics.get("capability_ceiling") is not False:
                raise ValueError("global capability_ceiling must be false")
            if semantics.get("research_bans_are_invalid") is not True:
                raise ValueError("research_bans_are_invalid must be true")
            for name, profile in profiles_payload.get("profiles", {}).items():
                if not isinstance(profile, dict):
                    raise ValueError(f"profile {name} must be an object")
                for field in (
                    "state", "capability_ceiling", "research_allowed",
                    "shadow_allowed", "activation_condition",
                    "review_at", "removal_criterion",
                ):
                    if field not in profile:
                        raise ValueError(f"profile {name} missing {field}")
                if profile["capability_ceiling"] is not False:
                    raise ValueError(
                        f"profile {name} created a capability ceiling"
                    )
        except Exception as exc:  # noqa: BLE001
            add_finding(
                findings, "critical",
                "policies/capability_profiles.json",
                "CAPABILITY_PROFILE_SEMANTICS_INVALID",
                f"{type(exc).__name__}: {exc}",
            )

    registry_path = repo / "policies" / "phase_scope_registry.json"
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            rules = registry["capability_expansion_rules"]
            required_true = (
                "platform_lock_in_requires_profile_scope",
                "broad_hold_requires_sunset",
                "fixed_limits_require_profile_and_override",
                "research_bans_are_invalid",
                "destination_policy_requires_successor_path",
                "historical_compatibility_cannot_govern_successors",
            )
            for field in required_true:
                if rules.get(field) is not True:
                    raise ValueError(f"registry rule {field} must be true")
            entries = registry.get("entries", {})
            for rel in ACTIVE_DESTINATION_DOCS:
                entry = entries.get(rel)
                if not isinstance(entry, dict):
                    raise ValueError(f"registry missing {rel}")
                for field in (
                    "scope_kind", "profile", "capability_ceiling",
                    "research_allowed", "shadow_allowed", "successor_path",
                    "production_activation_condition", "review_at",
                    "removal_criterion",
                ):
                    if field not in entry:
                        raise ValueError(f"{rel} missing {field}")
                if entry["capability_ceiling"] is not False:
                    raise ValueError(f"{rel} created a capability ceiling")
        except Exception as exc:  # noqa: BLE001
            add_finding(
                findings, "critical",
                "policies/phase_scope_registry.json",
                "PHASE_SCOPE_SUCCESSOR_METADATA_INVALID",
                f"{type(exc).__name__}: {exc}",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-head")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git worktree: {repo}")

    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()
    if args.expected_head and head != args.expected_head:
        raise SystemExit(
            f"HEAD mismatch: expected {args.expected_head}, got {head}"
        )

    paths = git_paths(repo)
    findings: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for rel in paths:
        path = repo / rel
        if not path.is_file():
            add_finding(
                findings, "critical", rel,
                "TRACKED_FILE_MISSING", "git ls-files path is absent",
            )
            continue
        lowered_parts = {part.lower() for part in Path(rel).parts}
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_PATH_SUFFIXES:
            add_finding(
                findings, "critical", rel,
                "FORBIDDEN_ARTIFACT", f"forbidden suffix {suffix}",
            )
        if lowered_parts & FORBIDDEN_PATH_PARTS:
            add_finding(
                findings, "critical", rel,
                "GENERATED_CACHE", "generated cache/editor artifact",
            )
        if (
            path.name == ".env"
            or (
                path.name.startswith(".env.")
                and path.name != ".env.example"
            )
        ):
            add_finding(
                findings, "critical", rel,
                "ENVIRONMENT_SECRET_FILE",
                "environment secret file committed",
            )
        data = path.read_bytes()
        category = classify(path)
        parse_status = "binary_or_unknown"
        if category != "binary_or_unknown":
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                add_finding(
                    findings, "critical", rel,
                    "UTF8_DECODE_FAILURE", str(exc),
                )
                parse_status = "failed"
            else:
                parse_status = inspect_text(
                    rel=rel,
                    text=text,
                    category=category,
                    findings=findings,
                )
        files.append(
            {
                "path": rel,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "category": category,
                "parse_status": parse_status,
            }
        )

    # Preserve the released Phase 4 acceptance markers.
    required = {
        "README.md": "P4.10 operational live-public-information closure",
        "docs/ROADMAP.md": "P4.10 operational live-public-information closure",
        "docs/CAPABILITY_CATALOG.md": "P4.10 LIVE ACCEPTANCE",
        "docs/PHASE_4_POST_PHASE_AUDIT.md":
            "operational live-public-information closure remains required",
        "docs/PHASE_BOUNDARY_AUDIT_STANDARD.md":
            "Every top-level phase ends with an adversarial audit",
        "docs/decisions/ADR-009-phase4-live-public-information-closure.md":
            "Phase 5 is blocked until P4.10",
        "policies/information_live_provider_acceptance_policy.json":
            '"phase5_start_gate": "blocked_until_p4_10_approved"',
    }
    for rel, marker in required.items():
        require_marker(
            repo=repo, rel=rel, marker=marker, findings=findings
        )

    inspect_destination_policy(repo, findings)

    counts = {
        severity: sum(
            1 for item in findings if item["severity"] == severity
        )
        for severity in ("critical", "major", "advisory")
    }
    report = {
        "schema_version": 2,
        "repository_head": head,
        "tracked_file_count": len(paths),
        "files": files,
        "findings": findings,
        "counts": counts,
        "approved": counts["critical"] == 0,
        "capability_ceiling": False,
        "private_content_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "phase-boundary repository audit: "
        f"{len(paths)} tracked files; "
        f"{counts['critical']} critical; "
        f"{counts['major']} major; "
        f"{counts['advisory']} advisory"
    )
    print(f"report={output}")
    return 0 if report["approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
