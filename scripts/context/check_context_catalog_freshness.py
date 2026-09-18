#!/usr/bin/env python3
"""Detect whether the branch-aware A.L.I.C.E. context catalog is stale."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SOURCE_BRANCH = "alice-eipm-v1-build"
RESEARCH_BRANCH = "research/graphify-context-substrate"
CATALOG_DIR = Path("research/graphify-context/catalog")


def git(*args: str) -> str:
    p=subprocess.run(
        ["git",*args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.stdout


def relevant(name: str) -> bool:
    # Fail open on branch coverage, fail closed on authority. Every pushed
    # non-temporary ALICE branch can carry lessons, receipts, telemetry, or
    # superseded decisions. The research substrate and tmp-* branches are the
    # only deliberate exclusions.
    return name != RESEARCH_BRANCH and not name.startswith("tmp-")


def actual_heads() -> dict[str,str]:
    raw=git(
        "for-each-ref",
        "--format=%(refname:strip=3)%00%(objectname)",
        "refs/remotes/origin",
    )
    out={}
    for line in raw.splitlines():
        if "\x00" not in line:
            continue
        name,sha=line.split("\x00",1)
        if name=="HEAD" or not relevant(name):
            continue
        out[name]=sha
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="research/graphify-context/CATALOG_FRESHNESS.json",
    )
    ap.add_argument("--github-env")
    ap.add_argument("--fail-on-drift",action="store_true")
    args=ap.parse_args()

    actual=actual_heads()
    branch_path=CATALOG_DIR/"BRANCH_CATALOG.json"
    status_path=CATALOG_DIR/"CATALOG_STATUS.json"

    issues=[]
    recorded={}
    catalog_source=None
    if not branch_path.exists() or not status_path.exists():
        issues.append({
            "kind":"catalog_missing",
            "branch_catalog_exists":branch_path.exists(),
            "catalog_status_exists":status_path.exists(),
        })
    else:
        branch_payload=json.loads(branch_path.read_text(encoding="utf-8"))
        status=json.loads(status_path.read_text(encoding="utf-8"))
        catalog_source=status.get("source_commit")
        for row in branch_payload.get("branches",[]):
            name=row.get("branch")
            if not name or not relevant(name):
                continue
            recorded[name]=row.get("head")

        for name,sha in sorted(actual.items()):
            if name not in recorded:
                issues.append({
                    "kind":"new_context_branch",
                    "branch":name,
                    "actual_head":sha,
                })
            elif recorded[name] != sha:
                issues.append({
                    "kind":"branch_head_moved",
                    "branch":name,
                    "recorded_head":recorded[name],
                    "actual_head":sha,
                })

        for name,sha in sorted(recorded.items()):
            if name not in actual:
                issues.append({
                    "kind":"catalog_branch_missing_remotely",
                    "branch":name,
                    "recorded_head":sha,
                })

    live_source=actual.get(SOURCE_BRANCH)
    if live_source is None:
        issues.append({
            "kind":"source_branch_missing",
            "branch":SOURCE_BRANCH,
        })
    elif catalog_source is not None and catalog_source != live_source:
        issues.append({
            "kind":"catalog_source_moved",
            "branch":SOURCE_BRANCH,
            "recorded_head":catalog_source,
            "actual_head":live_source,
        })

    result={
        "schema":"alice-context-catalog-freshness-v1",
        "fresh":not issues,
        "source_branch":SOURCE_BRANCH,
        "live_source_commit":live_source,
        "catalog_source_commit":catalog_source,
        "actual_context_branch_count":len(actual),
        "recorded_context_branch_count":len(recorded),
        "issues":issues,
    }
    Path(args.output).write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8",
    )

    if args.github_env:
        with open(args.github_env,"a",encoding="utf-8") as handle:
            handle.write(
                "REFRESH_CONTEXT_CATALOG="
                + ("false" if result["fresh"] else "true")
                + "\n"
            )
            if live_source:
                handle.write("LIVE_SOURCE_COMMIT="+live_source+"\n")

    print(json.dumps(result,ensure_ascii=False))
    if args.fail_on_drift and not result["fresh"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
