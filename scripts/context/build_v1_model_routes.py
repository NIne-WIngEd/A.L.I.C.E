#!/usr/bin/env python3
"""Build or query exact-source V1 model routes beside the Graphify N0 graph.

This is a navigation index only. It neither extracts a V1 code graph nor
promotes branch documents, generated summaries, or test output to authority.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


OUTPUT = Path("research/graphify-context/V1_MODEL_ROUTES.json")
ROUTES = {
    "foundation": {
        "branch": "research/gemma4-v1-source-derivation-20260930",
        "files": {
            "docs/GEMMA4_V1_SOURCE_AND_DERIVATIVE_CONTRACT.md": "source custody and derivative contract",
            "src/alice_foundation/gemma4_v1.py": "clone, mutation, and derivative verification",
            "src/alice_foundation/gemma4_inventory.py": "source tensor inventory",
            "tests/foundation/test_gemma4_v1.py": "foundation contract tests",
        },
    },
    "mfm": {
        "branch": "research/mfm-v1-licensed-base-20260930",
        "files": {
            "docs/MFM_V1_LICENSED_BASE_DIRECTION_2026-09-30.md": "V1 source direction and qualification limits",
            "docs/MFM_V1_SPECIALIST_TRAINING_PATH_2026-09-30.md": "MFM specialist training path",
            "docs/MFM_V1_ROLE_BOUNDARY_DIAGNOSTIC_2026-09-30.md": "paired behavior and role boundary diagnostic",
            "docs/MFM_FORMATION_CONTRACT_V1_6_ADMISSION_2026-09-30.md": "full-role formation contract and admission gap",
            "src/cognitive_kernel/formation_v1_specialist.py": "separate formation model architecture",
            "src/cognitive_kernel/formation_semantics_v16.py": "versioned sensitivity, episode, relationship and mission semantics",
            "scripts/mfm/train_v1_formation_specialist.py": "training and processor preflight",
            "scripts/mfm/run_v1_formation_specialist.py": "trained and seeded control inference",
            "scripts/mfm/verify_gemma4_pretrained.py": "historical MFM source receipt schema",
            "scripts/mfm/run_gemma4_base_behavior.py": "historical baseline requiring MFM source receipt",
            "scripts/mfm/run_gemma4_prepared_base_behavior.py": "prepared source public diagnostic",
            "scripts/mfm/qualify_v1_role_boundary.py": "role boundary comparison",
        },
    },
    "fbm": {
        "branch": "research/fbm-v1-licensed-base-20260930",
        "files": {
            "training/fbm/base_assembly/gemma4_12b_mfm_v1.seed.json": "FBM ordered replay seed",
            "training/fbm/base_assembly/mfm_v1_training_role_coverage_20260930.json": "hash-bound training-role coverage audit",
            "docs/fable-builder/FBM_V1_GEMMA4_BASE_DISSECTION_AND_ROLE_RECIPE_2026-09-30.md": "FBM source and role recipe",
            "docs/fable-builder/traces/FBM_TRACE_20260930_GEMMA4_MFM_REPLAY_SEED.jsonl": "FBM process trace",
            "docs/fable-builder/traces/FBM_TRACE_20260930_GEMMA4_MFM_PUBLICATION_REBIND.jsonl": "published cross-branch lineage",
            "docs/fable-builder/traces/FBM_TRACE_20260930_GEMMA4_CLONE_FIRST_FORMATION_SEED.jsonl": "clone-first decision trace",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_FULL_ROLE_READINESS_AUDIT.jsonl": "full-fit stop and corpus audit trace",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_V15_PROBE_ONLY_CONTRACT.jsonl": "v1.5 trainer probe-only correction",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_V16_PUBLIC_CONTRACT_PIN.jsonl": "published v1.6 contract pin",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_PUBLIC_DOCS_CORRECTION_PIN.jsonl": "matched control and exact public MFM pin",
        },
    },
}


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return p.stdout.strip()


def live_heads() -> dict[str, str]:
    branches = [route["branch"] for route in ROUTES.values()]
    raw = git("ls-remote", "--heads", "origin", *branches)
    heads = {ref.removeprefix("refs/heads/"): oid for oid, ref in
             (line.split("\t") for line in raw.splitlines())}
    missing = sorted(set(branches) - set(heads))
    if missing:
        raise RuntimeError(f"V1 branch missing on origin: {missing}")
    return heads


def build() -> dict:
    live = live_heads()
    areas = {}
    for area, route in ROUTES.items():
        branch = route["branch"]
        local = git("rev-parse", f"refs/remotes/origin/{branch}")
        if local != live[branch]:
            raise RuntimeError(f"STALE_LOCAL_REF {branch}: local={local} live={live[branch]}; fetch first")
        files = []
        for path, purpose in route["files"].items():
            row = git("ls-tree", local, "--", path)
            if not row:
                raise RuntimeError(f"MISSING_SOURCE_FILE {branch}@{local}:{path}")
            metadata, name = row.split("\t", 1)
            mode, kind, blob = metadata.split()
            if kind != "blob" or name != path:
                raise RuntimeError(f"INVALID_SOURCE_FILE {branch}@{local}:{path}")
            files.append({"path": path, "purpose": purpose, "blob_sha": blob})
        areas[area] = {"branch": branch, "commit": local, "files": files}
    return {
        "schema": "alice.graphify.v1-model-routes.v1",
        "authority": "navigation-only",
        "graph_status": "existing Graphify code graph is N0-only; no V1 graph built",
        "qualification_status": "no model qualification can be inferred from this index",
        "areas": areas,
    }


def encoded(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--query", metavar="TEXT")
    args = ap.parse_args()
    expected = build()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(encoded(expected), encoding="utf-8")
        print(f"WROTE {OUTPUT}")
        return
    if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != encoded(expected):
        raise SystemExit("STALE_V1_ROUTES: regenerate from fetched exact live branch heads")
    if args.check:
        print("V1_ROUTES_FRESH: all three live source branches and file blobs match")
        return
    words = args.query.lower().split()
    matches = [
        {"area": area, "branch": route["branch"], "commit": route["commit"], **file}
        for area, route in expected["areas"].items()
        for file in route["files"]
        if all(word in f'{area} {file["purpose"]} {file["path"]}'.lower() for word in words)
    ]
    print(json.dumps(matches[:12], indent=2))


if __name__ == "__main__":
    main()
