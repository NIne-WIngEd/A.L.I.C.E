from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET = "src/alice_information/grounding_policy.py"
MILESTONE = "P4.5a"


def register(active_repo: Path, registry_path: Path) -> bool:
    target = active_repo / TARGET
    if not target.exists():
        print(f"active milestone scope skipped; file not present: {TARGET}")
        return False
    text = target.read_text(encoding="utf-8")
    required = (
        MILESTONE,
        "must remain disabled",
        "External actions must remain disabled",
        "Memory writes must remain disabled",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ValueError(
            "Refusing to register the P4.5a grounding scope because expected "
            f"phase-local guards are missing: {missing}"
        )

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("phase scope registry entries must be an object")
    entries[TARGET] = {
        "scope_kind": "active_milestone_guard",
        "capability_ceiling": False,
        "profile": "information.phase4.grounding",
        "milestone": MILESTONE,
        "applies_to_changed_lines": True,
        "changed_line_rule_codes": ["MUST_REMAIN_DISABLED"],
        "successor_runtime": "src/alice_evolution/capability_runtime.py",
        "reason": (
            "The P4.5a grounding policy is intentionally read-only and side-effect-free "
            "inside this milestone. It does not prohibit later information, memory, tool, "
            "or action profiles."
        ),
        "sunset_condition": (
            "Replace the hard-coded P4.5a guard with a selected capability profile when "
            "grounding is integrated with the Experience Ledger and action-capable runtimes."
        ),
        "registered_by": "alice-mega-architecture-migration-v1.3",
    }
    payload["version"] = "1.3.0"
    payload["migration_id"] = "alice-mega-architecture-migration-v1.3"
    payload["registered_file_count"] = len(entries)
    registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"registered active milestone scope: {TARGET} ({MILESTONE})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-repo", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    register(args.active_repo.resolve(), args.registry.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
