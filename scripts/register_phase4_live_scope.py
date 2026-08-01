#!/usr/bin/env python3
"""Idempotently register additive P4.10 scope, version, and status blocks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

P410A_FILES = (
    "docs/PHASE_4_LIVE_PROVIDER_CONFIGURATION.md",
    "policies/information_live_provider_runtime_policy.json",
    "scripts/register_phase4_live_scope.py",
    "scripts/run_phase4_live_provider_preflight.py",
    "src/alice_information/brave_search.py",
    "src/alice_information/brave_search_live.py",
    "src/alice_information/live_fetch_provider.py",
    "src/alice_information/live_provider_config.py",
    "src/alice_information/live_provider_contracts.py",
    "src/alice_information/live_provider_policy.py",
    "src/alice_information/live_provider_registry.py",
    "tests/phase4/_information_live_provider_helpers.py",
    "tests/phase4/test_information_brave_search.py",
    "tests/phase4/test_information_brave_search_live.py",
    "tests/phase4/test_information_live_fetch_provider.py",
    "tests/phase4/test_information_live_provider_config.py",
    "tests/phase4/test_information_live_provider_contracts.py",
    "tests/phase4/test_information_live_provider_policy.py",
    "tests/phase4/test_information_live_provider_preflight_script.py",
    "tests/phase4/test_information_live_provider_registry.py",
)
P410B_FILES = (
    "docs/PHASE_4_LIVE_RESEARCH_EXECUTION.md",
    "policies/information_live_research_policy.json",
    "scripts/run_phase4_live_research.py",
    "src/alice_conversation/state_schema.py",
    "src/alice_conversation/state_store.py",
    "src/alice_information/live_claims.py",
    "src/alice_information/live_research.py",
    "src/alice_information/live_research_policy.py",
    "tests/phase3/test_conversation_state_store.py",
    "tests/phase3/test_conversation_web_source_state_migration.py",
    "tests/phase3/test_conversation_reference_identity_migration.py",
    "tests/phase4/_information_live_research_helpers.py",
    "tests/phase4/test_information_live_claims.py",
    "tests/phase4/test_information_live_research.py",
    "tests/phase4/test_information_live_research_adversarial.py",
    "tests/phase4/test_information_live_research_policy.py",
    "tests/phase4/test_information_live_research_script.py",
)
P410C_FILES = (
    "benchmarks/phase4/information_live_acceptance_v1.json",
    "docs/PHASE_4_LIVE_OPERATIONAL_RELEASE_REPORT.md",
    "policies/information_live_acceptance_release_policy.json",
    "scripts/run_phase4_live_information_acceptance.py",
    "src/alice_information/live_acceptance.py",
    "src/alice_information/live_acceptance_inspection.py",
    "tests/phase4/_information_live_acceptance_helpers.py",
    "tests/phase4/test_information_live_acceptance.py",
    "tests/phase4/test_information_live_acceptance_adversarial.py",
    "tests/phase4/test_information_live_acceptance_inspection.py",
    "tests/phase4/test_information_live_acceptance_script.py",
)


def _upsert_block(path: Path, *, marker: str, content: str) -> None:
    text = path.read_text(encoding="utf-8-sig")
    start = f"<!-- {marker} START -->"
    end = f"<!-- {marker} END -->"
    block = f"{start}\n{content.strip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        updated = pattern.sub(block, text, count=1)
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8", newline="\n")


def _status(milestone: str) -> tuple[str, str, str]:
    if milestone == "P4.10a":
        return (
            "0.16.0",
            "P4.10a live-provider foundation is complete. P4.10b live governed research execution remains active. Phase 5 remains blocked.",
            "P4.10a",
        )
    if milestone == "P4.10b":
        return (
            "0.17.0",
            "P4.10a–P4.10b are complete. P4.10c private live acceptance and exact-commit closure remains active. Phase 5 remains blocked.",
            "P4.10b",
        )
    return (
        "0.18.0",
        "P4.10 operational live-public-information closure is implemented. Phase 4 becomes operationally complete only after the private P4.10c record is approved and the exact audited tree is merged.",
        "P4.10c",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--milestone", choices=("P4.10a", "P4.10b", "P4.10c"), required=True
    )
    args = parser.parse_args()
    root = Path(args.repo).resolve(strict=True)
    version, status, completed = _status(args.milestone)
    selected = list(P410A_FILES)
    if args.milestone in {"P4.10b", "P4.10c"}:
        selected.extend(P410B_FILES)
    if args.milestone == "P4.10c":
        selected.extend(P410C_FILES)
    missing = [relative for relative in selected if not (root / relative).is_file()]
    if missing:
        raise SystemExit("Missing P4.10 files: " + ", ".join(missing))

    registry_path = root / "policies/phase_scope_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    entries = registry.get("entries")
    if not isinstance(entries, dict):
        raise SystemExit("phase_scope_registry.json has no entries object")
    def milestone_for(relative: str) -> str:
        if relative in P410A_FILES:
            return "P4.10a"
        if relative in P410B_FILES:
            return "P4.10b"
        if relative in P410C_FILES:
            return "P4.10c"
        raise SystemExit(f"Unclassified P4.10 scope file: {relative}")

    for relative in selected:
        file_milestone = milestone_for(relative)
        common = {
            "capability_ceiling": False,
            "profile": "information.phase4.live_public",
            "successor_runtime": "src/alice_evolution/capability_runtime.py",
            "registered_by": "phase4-p410-operational-live-closure",
        }
        if relative.startswith("tests/"):
            entries[relative] = {
                **common,
                "scope_kind": "compatibility_test",
                "reason": (
                    f"{file_milestone} tests bind the additive live PUBLIC profile "
                    "and cannot veto later authorized information capabilities."
                ),
            }
        elif relative.startswith("docs/"):
            entries[relative] = {
                **common,
                "scope_kind": "historical_or_phase_local",
                "reason": (
                    f"{file_milestone} documentation defines the current live PUBLIC "
                    "release profile, not the destination capability ceiling."
                ),
            }
        elif relative.startswith("policies/") or relative.startswith("benchmarks/"):
            entries[relative] = {
                **common,
                "scope_kind": "phase_local_compatibility",
                "reason": (
                    f"{file_milestone} policy/evaluation data is selected only for the "
                    "named live PUBLIC compatibility profile."
                ),
            }
        else:
            entries[relative] = {
                **common,
                "scope_kind": "active_milestone_guard",
                "milestone": file_milestone,
                "applies_to_changed_lines": True,
                "changed_line_rule_codes": [
                    "FALSE_CAPABILITY_ASSERTION",
                    "MUST_REMAIN_DISABLED",
                ],
                "reason": (
                    f"{file_milestone} is a bounded foreground PUBLIC live-research "
                    "profile. Its no-persistence, no-action, no-retry, no-fallback, and "
                    "no-background controls do not prohibit later authorized profiles."
                ),
                "sunset_condition": (
                    "Retain this exact Phase 4 live-public release profile for "
                    "reproducibility while successor capabilities are selected through "
                    "the shared capability runtime."
                ),
            }
    registry["registered_file_count"] = len(entries)
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    init_path = root / "src/alice_information/__init__.py"
    init_text = init_path.read_text(encoding="utf-8-sig")
    updated, count = re.subn(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{version}"',
        init_text,
        count=1,
    )
    if count != 1:
        raise SystemExit("alice_information version marker was not found exactly once")
    init_path.write_text(updated, encoding="utf-8", newline="\n")

    public_status = (
        f"## Additive Phase 4 live-public-information status\n\n"
        f"- Completed sub-milestone: **{completed}**\n"
        f"- Package profile: `alice_information {version}`\n"
        f"- {status}\n"
        "- P4.0–P4.9 and the P4.6a/P4.7a/P4.7b fixture profiles remain reproducible and unchanged.\n"
        "- No source persistence, Phase 5 storage, memory write, external action, recursive browse, retry, fallback, or background execution is activated."
    )
    for relative in (
        "README.md",
        "docs/ROADMAP.md",
        "docs/PHASE_4_WEB_INFORMATION_ARCHITECTURE.md",
        "docs/CAPABILITY_CATALOG.md",
    ):
        path = root / relative
        if path.is_file():
            _upsert_block(path, marker="P4.10 LIVE PUBLIC STATUS", content=public_status)
    print(
        f"milestone={args.milestone} registered={len(selected)} "
        f"version={version} entries={len(entries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
