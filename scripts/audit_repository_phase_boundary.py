from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
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


def git_paths(repo: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
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


def add_finding(findings: list[dict[str, Any]], severity: str, path: str, code: str, detail: str) -> None:
    findings.append(
        {"severity": severity, "path": path, "code": code, "detail": detail}
    )


def inspect_text(
    *,
    repo: Path,
    rel: str,
    text: str,
    category: str,
    findings: list[dict[str, Any]],
) -> str:
    parse_status = "not_applicable"

    if CONFLICT_PATTERN.search(text):
        add_finding(findings, "critical", rel, "MERGE_CONFLICT_MARKER", "merge conflict marker found")

    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            add_finding(findings, "critical", rel, f"SECRET_{name.upper()}", "high-confidence secret pattern found")

    trailing = sum(1 for line in text.splitlines() if line.endswith((" ", "\t")))
    if trailing:
        add_finding(findings, "advisory", rel, "TRAILING_WHITESPACE", f"{trailing} line(s)")

    try:
        if category == "python":
            ast.parse(text, filename=rel)
            parse_status = "passed"
        elif category == "json":
            json.loads(text)
            parse_status = "passed"
        elif category == "yaml":
            if yaml is None:
                add_finding(findings, "major", rel, "YAML_PARSER_UNAVAILABLE", "PyYAML is required")
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
        add_finding(findings, "critical", rel, "PARSE_FAILURE", f"{type(exc).__name__}: {exc}")
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
        add_finding(findings, "critical", rel, "REQUIRED_FILE_MISSING", "required controlling file missing")
        return
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        add_finding(findings, "critical", rel, "REQUIRED_MARKER_MISSING", marker)


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
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.expected_head and head != args.expected_head:
        raise SystemExit(f"HEAD mismatch: expected {args.expected_head}, got {head}")

    paths = git_paths(repo)
    findings: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    for rel in paths:
        path = repo / rel
        if not path.is_file():
            add_finding(findings, "critical", rel, "TRACKED_FILE_MISSING", "git ls-files path is absent")
            continue

        lowered_parts = {part.lower() for part in Path(rel).parts}
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_PATH_SUFFIXES:
            add_finding(findings, "critical", rel, "FORBIDDEN_ARTIFACT", f"forbidden suffix {suffix}")
        if lowered_parts & FORBIDDEN_PATH_PARTS:
            add_finding(findings, "critical", rel, "GENERATED_CACHE", "generated cache/editor artifact")
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            add_finding(findings, "critical", rel, "ENVIRONMENT_SECRET_FILE", "environment secret file committed")

        data = path.read_bytes()
        category = classify(path)
        parse_status = "binary_or_unknown"

        if category != "binary_or_unknown":
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                add_finding(findings, "critical", rel, "UTF8_DECODE_FAILURE", str(exc))
                parse_status = "failed"
            else:
                parse_status = inspect_text(
                    repo=repo,
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

    required = {
        "README.md": "P4.10 operational live-public-information closure",
        "docs/ROADMAP.md": "P4.10 operational live-public-information closure",
        "docs/CAPABILITY_CATALOG.md": "P4.10 LIVE ACCEPTANCE",
        "docs/PHASE_4_POST_PHASE_AUDIT.md": "operational live-public-information closure remains required",
        "docs/PHASE_BOUNDARY_AUDIT_STANDARD.md": "Every top-level phase ends with an adversarial audit",
        "docs/decisions/ADR-009-phase4-live-public-information-closure.md": "Phase 5 is blocked until P4.10",
        "policies/information_live_provider_acceptance_policy.json": '"phase5_start_gate": "blocked_until_p4_10_approved"',
    }
    for rel, marker in required.items():
        require_marker(repo=repo, rel=rel, marker=marker, findings=findings)

    counts = {
        severity: sum(1 for item in findings if item["severity"] == severity)
        for severity in ("critical", "major", "advisory")
    }
    report = {
        "schema_version": 1,
        "repository_head": head,
        "tracked_file_count": len(paths),
        "files": files,
        "findings": findings,
        "counts": counts,
        "approved": counts["critical"] == 0,
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
