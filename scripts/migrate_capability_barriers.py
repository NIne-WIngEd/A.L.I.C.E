#!/usr/bin/env python3
"""One-time migration of legacy capability barriers into scoped compatibility records.

This script is intentionally separate from CI. It may register the existing reviewed
baseline during the Roadmap 2 / Governance 1 migration. Normal CI only audits and may
not silently register new restrictions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_capability_barriers import Finding, audit

MARKER = "A.L.I.C.E. CAPABILITY SCOPE"


def _profile_for(path: str) -> str:
    lowered = path.lower()
    if "information" in lowered or "phase4" in lowered or "phase_4" in lowered:
        return "information.phase4.foundation"
    if "orchestration" in lowered:
        return "orchestration.phase3.compatibility"
    if "conversation" in lowered or "phase3" in lowered or "phase_3" in lowered:
        return "conversation.phase3.compatibility"
    if "phase2" in lowered or "phase_2" in lowered or "/memory" in lowered:
        return "memory.phase2.compatibility"
    if "phase1" in lowered or "phase_1" in lowered or "/vault" in lowered:
        return "vault.phase1.compatibility"
    return "legacy.release.compatibility"


def _scope_for(path: str) -> str:
    lowered = path.lower()
    if lowered.startswith(("legacy/", "archive/", "docs/decisions/adr-001", "docs/decisions/adr-002")):
        return "historical_superseded"
    if lowered.startswith("tests/"):
        return "compatibility_test"
    if lowered.startswith("docs/") and any(token in lowered for token in ("phase_", "phase3", "phase4", "p3.", "p4.")):
        return "historical_or_phase_local"
    return "phase_local_compatibility"


def _banner(path: str, scope: str, profile: str) -> str:
    if path.endswith(".md"):
        return (
            f"> **{MARKER}:** `{scope}`; `capability_ceiling=false`; "
            f"profile `{profile}`. Restrictions below reproduce a released or historical "
            "configuration and do not limit successor A.L.I.C.E. capabilities.\n\n"
        )
    if path.endswith((".py", ".ps1", ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg")):
        prefix = "#"
        return (
            f"{prefix} {MARKER}: scope={scope}; capability_ceiling=false; "
            f"profile={profile}.\n"
            f"{prefix} This file preserves a released compatibility behavior; it is not a project-wide veto.\n"
        )
    return ""


def _annotate(repo: Path, relative: str, scope: str, profile: str) -> bool:
    path = repo / relative
    if not path.exists() or path.suffix.lower() == ".json":
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if MARKER in text[:1200]:
        return False
    banner = _banner(relative, scope, profile)
    if not banner:
        return False
    if text.startswith("#!"):
        first, separator, remainder = text.partition("\n")
        updated = first + separator + banner + remainder
    else:
        updated = banner + text
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def _entry(path: str, findings: list[Finding]) -> dict[str, object]:
    scope = _scope_for(path)
    profile = _profile_for(path)
    return {
        "scope_kind": scope,
        "capability_ceiling": False,
        "profile": profile,
        "successor_runtime": "src/alice_evolution/capability_runtime.py",
        "barrier_codes": sorted({finding.code for finding in findings}),
        "reason": (
            "Preserves reproducible behavior or tests from an earlier release while "
            "broader capability is enabled through named profiles and missions."
        ),
        "registered_by": "project-wide-capability-unblocking-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--register-existing", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--acknowledge-bulk-registration", action="store_true")
    args = parser.parse_args()
    if not args.register_existing or not args.acknowledge_bulk_registration:
        parser.error("Bulk registration is exceptional and requires both --register-existing and --acknowledge-bulk-registration after human review.")

    repo = args.repo.resolve()
    registry_path = repo / "policies" / "phase_scope_registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("phase scope registry entries must be an object")

    findings = audit(repo)
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        if finding.disposition == "unresolved_active_barrier" or finding.disposition == "historical_unregistered":
            grouped[finding.path].append(finding)

    annotated = 0
    for path, path_findings in sorted(grouped.items()):
        item = _entry(path, path_findings)
        entries[path] = item
        if args.annotate and _annotate(
            repo,
            path,
            str(item["scope_kind"]),
            str(item["profile"]),
        ):
            annotated += 1

    payload["generated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload["registered_file_count"] = len(entries)
    registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"registered {len(grouped)} files; annotated {annotated} text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
