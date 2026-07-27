from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

BASELINE_ROOTS = {
    "src/alice_vault/": (
        "vault.phase1.compatibility",
        "Released Phase 1 source remains reproducible while the shared kernel provides the successor path.",
    ),
    "src/alice_memory/": (
        "memory.phase2.compatibility",
        "Released Phase 2 source remains reproducible while lifelong-learning successors evolve separately.",
    ),
    "src/alice_conversation/": (
        "conversation.phase3.compatibility",
        "Released Phase 3 source remains reproducible while integrated conversation profiles provide the successor path.",
    ),
    "src/alice_information/": (
        "information.phase4.foundation",
        "Released Phase 4 source remains reproducible while live, authorized, and learning-aware profiles evolve separately.",
    ),
}


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def register(repo: Path, registry_path: Path, ref: str) -> int:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("phase scope registry entries must be an object")

    baseline_commit = _run(repo, "rev-parse", ref)
    paths = _run(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        ref,
        "--",
        *[root.rstrip("/") for root in BASELINE_ROOTS],
    ).splitlines()

    added = 0
    for raw in sorted(set(paths)):
        relative = raw.strip().replace("\\", "/")
        if not relative:
            continue
        selected: tuple[str, str] | None = None
        for prefix, metadata in BASELINE_ROOTS.items():
            if relative.startswith(prefix):
                selected = metadata
                break
        if selected is None or relative in entries:
            continue
        profile, reason = selected
        entries[relative] = {
            "scope_kind": "released_baseline_compatibility",
            "capability_ceiling": False,
            "profile": profile,
            "successor_runtime": "src/alice_evolution/capability_runtime.py",
            "baseline_commit": baseline_commit,
            "reason": reason,
            "registered_by": "alice-mega-architecture-migration-v1.3",
        }
        added += 1

    payload["version"] = "1.3.0"
    payload["migration_id"] = "alice-mega-architecture-migration-v1.3"
    payload["released_baseline_commit"] = baseline_commit
    payload["registered_file_count"] = len(entries)
    registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return added


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()

    repo = args.repo.resolve()
    registry = args.registry.resolve()
    added = register(repo, registry, args.ref)
    print(f"registered released baseline compatibility scopes: {added} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
