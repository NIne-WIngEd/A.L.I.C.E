#!/usr/bin/env python3
"""Build deterministic A.L.I.C.E. context catalogs.

This script does not decide truth. It records source pointers, branch lineage,
document metadata, failure lessons, research surfaces, and supersession hints so
an agent can retrieve the smallest authoritative source set for a task.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

SOURCE_BRANCH = "alice-eipm-v1-build"
OUT = Path("research/graphify-context/catalog")
MAX_TEXT_BYTES = 1_500_000

TEXT_EXTS = {
    ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".tsv",
    ".csv", ".sh", ".sbatch", ".ps1", ".py",
}

DOC_EXTS = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".tsv"}

KNOWLEDGE_BRANCH_PATTERNS = (
    r"^main$",
    r"^alice-context$",
    r"^fable-builder-model$",
    r"^alice-mc10[bc]-live$",
    r"^alice-eipm-v1-",
    r"^docs/",
    r"^planning/",
    r"^feat/memory",
    r"^fix/governance",
)

FAILURE_TERMS = (
    "failure", "failed", "error", "repair", "incident", "forensic", "recovery",
    "fallback", "magnolia", "kaggle", "udocker", "powershell", "hotfix",
    "network", "dns", "egress", "smoke", "runtime-results",
)

RESEARCH_TERMS = (
    "frontier", "competitor", "research", "external_system", "external systems",
    "architecture research", "benchmark", "model review", "literature",
)

SUPERSESSION_PATTERNS = (
    re.compile(r"\bsupersed(?:e|es|ed|ing)\b", re.I),
    re.compile(r"\breplaced?\s+by\b", re.I),
    re.compile(r"\bno\s+longer\b", re.I),
    re.compile(r"\bobsolete\b", re.I),
    re.compile(r"\bdeprecated\b", re.I),
    re.compile(r"\bretired\b", re.I),
    re.compile(r"\bdo\s+not\s+use\b", re.I),
)

STATUS_PATTERNS = {
    "status": re.compile(r"^\s*(?:\*\*)?status(?:\*\*)?\s*:\s*(.+?)\s*$", re.I),
    "supersession": re.compile(r"^\s*(?:\*\*)?supersession(?:\*\*)?\s*:\s*(.+?)\s*$", re.I),
    "owner_constraint": re.compile(r"^\s*(?:\*\*)?owner constraint(?:\*\*)?\s*:\s*(.+?)\s*$", re.I),
    "decision": re.compile(r"^\s*(?:\*\*)?decision(?:\*\*)?\s*:\s*(.+?)\s*$", re.I),
}


def git(*args: str, check: bool = True) -> str:
    p = subprocess.run(
        ["git", *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.stdout


def remote_ref(branch: str) -> str:
    return f"refs/remotes/origin/{branch}"


def list_remote_branches() -> list[str]:
    raw = git(
        "for-each-ref",
        "--format=%(refname:strip=3)",
        "refs/remotes/origin",
    )
    return sorted(
        x.strip()
        for x in raw.splitlines()
        if x.strip() and x.strip() != "HEAD"
    )


def branch_sha(branch: str) -> str:
    return git("rev-parse", remote_ref(branch)).strip()


def relation(source: str, other: str) -> dict:
    left, right = git(
        "rev-list", "--left-right", "--count",
        f"{remote_ref(source)}...{remote_ref(other)}",
    ).strip().split()
    behind = int(left)
    ahead = int(right)
    try:
        merge_base = git(
            "merge-base", remote_ref(source), remote_ref(other)
        ).strip()
    except subprocess.CalledProcessError:
        merge_base = None

    if merge_base is None:
        status = "unrelated"
    elif ahead and behind:
        status = "diverged"
    elif ahead:
        status = "ahead"
    elif behind:
        status = "behind"
    else:
        status = "identical"
    return {
        "status": status,
        "ahead": ahead,
        "behind": behind,
        "merge_base": merge_base,
    }


def changed_paths(source: str, other: str, merge_base: str | None) -> list[str]:
    if source == other:
        return []
    if merge_base:
        raw = git(
            "diff", "--name-only",
            f"{remote_ref(source)}...{remote_ref(other)}",
        )
    else:
        # Histories with no merge-base are kept explicitly separate. Compare
        # endpoint trees only for inventory; do not imply shared lineage.
        raw = git(
            "diff", "--name-only",
            remote_ref(source), remote_ref(other),
        )
    return [x for x in raw.splitlines() if x.strip()]


def ls_tree(ref: str) -> list[dict]:
    raw = git("ls-tree", "-r", "-l", ref)
    rows = []
    for line in raw.splitlines():
        # <mode> <type> <sha> <size>\t<path>
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        size = None if parts[3] == "-" else int(parts[3])
        rows.append({"path": path, "sha": parts[2], "size": size})
    return rows


def safe_text(ref: str, path: str, size: int | None) -> str | None:
    if size is not None and size > MAX_TEXT_BYTES:
        return None
    ext = Path(path).suffix.lower()
    if ext not in TEXT_EXTS:
        return None
    try:
        return git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None


def classify_path(path: str) -> str:
    p = path.lower()
    if "/runtime-results/" in p or "receipt" in p:
        return "runtime_receipt"
    if p.startswith("docs/decisions/"):
        return "decision_record"
    if "chat-context/" in p:
        return "working_context"
    if p.startswith("policies/"):
        return "policy"
    if p.startswith("configs/"):
        return "configuration_state"
    if p.startswith("training/"):
        return "training_material"
    if p.startswith("evaluation/") or p.startswith("benchmarks/"):
        return "evaluation"
    if p.startswith("tests/"):
        return "test"
    if p.startswith("src/") or p.startswith("scripts/"):
        return "implementation"
    if p == "readme.md":
        return "project_readme"
    if p.startswith("docs/"):
        return "design_document"
    return "repository_file"


def extract_metadata(text: str, path: str) -> dict:
    title = None
    headings: list[str] = []
    status_fields: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                if title is None:
                    title = heading
                if len(headings) < 16:
                    headings.append(heading)
        for key, pattern in STATUS_PATTERNS.items():
            if key not in status_fields:
                m = pattern.match(raw)
                if m:
                    status_fields[key] = m.group(1).strip().strip("*")
    if not title:
        title = Path(path).name
    return {
        "title": title,
        "headings": headings,
        "declared": status_fields,
    }


def keyword_hits(path: str, text: str, terms: Iterable[str]) -> list[str]:
    hay = f"{path}\n{text[:250_000]}".lower()
    return sorted({term for term in terms if term in hay})


def supersession_hits(text: str, limit: int = 12) -> list[dict]:
    hits: list[dict] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if any(p.search(raw) for p in SUPERSESSION_PATTERNS):
            hits.append({"line": number, "text": raw.strip()[:500]})
            if len(hits) >= limit:
                break
    return hits


def branch_is_knowledge_surface(name: str) -> bool:
    return any(re.search(pattern, name) for pattern in KNOWLEDGE_BRANCH_PATTERNS)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    branches = list_remote_branches()
    if SOURCE_BRANCH not in branches:
        raise RuntimeError(f"Missing origin/{SOURCE_BRANCH}")

    source_sha = branch_sha(SOURCE_BRANCH)
    branch_rows = []
    for name in branches:
        if name.startswith("tmp-") or name == "research/graphify-context-substrate":
            continue
        rel = relation(SOURCE_BRANCH, name)
        changed = changed_paths(SOURCE_BRANCH, name, rel["merge_base"])
        branch_rows.append({
            "branch": name,
            "head": branch_sha(name),
            **rel,
            "changed_file_count_vs_source_merge_base": len(changed),
            "knowledge_surface": branch_is_knowledge_surface(name),
        })

    branch_rows.sort(key=lambda x: x["branch"])
    write_json(
        OUT / "BRANCH_CATALOG.json",
        {
            "schema": "alice-context-branch-catalog-v1",
            "source_branch": SOURCE_BRANCH,
            "source_commit": source_sha,
            "branches": branch_rows,
        },
    )

    source_files = ls_tree(remote_ref(SOURCE_BRANCH))

    # Resolve the newest authoritative N0 latent-stage state. Older state files
    # remain indexed as history, but this pointer gives agents a bounded current
    # mission entry point without treating filename recency as truth elsewhere.
    state_candidates = []
    state_re = re.compile(r"^configs/eipm/n0/alice_n0_latent_pool_stage_state_v(\d+)\.(\d+)\.json$")
    for item in source_files:
        match = state_re.match(item["path"])
        if not match:
            continue
        text = safe_text(remote_ref(SOURCE_BRANCH), item["path"], item.get("size"))
        if text is None:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if payload.get("authoritative_for_current_latent_stage") is not True:
            continue
        state_candidates.append(((int(match.group(1)), int(match.group(2))), item, payload))

    active_n0_state = None
    if state_candidates:
        _version, item, payload = max(state_candidates, key=lambda row: row[0])
        repair = payload.get("relation_endpoint_repair_v0_2") or payload.get("evidence_selector_repair_v0_1") or {}
        active_n0_state = {
            "schema": "alice-context-active-n0-state-v1",
            "source_branch": SOURCE_BRANCH,
            "source_commit": source_sha,
            "source_path": item["path"],
            "source_blob_sha": item["sha"],
            "status": payload.get("status"),
            "n0_complete": payload.get("n0_complete"),
            "supersedes": payload.get("supersedes"),
            "pass_next_action": repair.get("pass_next_action"),
            "fail_next_action": repair.get("fail_next_action"),
            "scale_decision": payload.get("scale_decision"),
            "anti_loop_policy": payload.get("anti_loop_policy"),
            "selection_rule": "highest version among source-branch state files explicitly marked authoritative_for_current_latent_stage=true",
            "authority": "pointer_to_original_source",
        }
        write_json(OUT / "ACTIVE_N0_STATE.json", active_n0_state)

    source_rows = []
    for item in source_files:
        source_rows.append({
            **item,
            "branch": SOURCE_BRANCH,
            "category": classify_path(item["path"]),
            "extension": Path(item["path"]).suffix.lower(),
        })
    write_jsonl(OUT / "SOURCE_FILE_CATALOG.jsonl", source_rows)

    doc_rows: list[dict] = []
    failure_rows: list[dict] = []
    research_rows: list[dict] = []
    supersession_rows: list[dict] = []

    def add_doc(branch: str, ref: str, item: dict, branch_scope: str) -> None:
        path = item["path"]
        if Path(path).suffix.lower() not in DOC_EXTS:
            return
        text = safe_text(ref, path, item.get("size"))
        if text is None:
            return
        meta = extract_metadata(text, path)
        row = {
            "branch": branch,
            "branch_scope": branch_scope,
            "path": path,
            "blob_sha": item["sha"],
            "size": item.get("size"),
            "category": classify_path(path),
            **meta,
        }
        doc_rows.append(row)

        failures = keyword_hits(path, text, FAILURE_TERMS)
        if failures:
            failure_rows.append({
                **row,
                "matched_terms": failures,
            })

        research = keyword_hits(path, text, RESEARCH_TERMS)
        if research:
            research_rows.append({
                **row,
                "matched_terms": research,
            })

        sup = supersession_hits(text)
        if sup:
            supersession_rows.append({
                **row,
                "supersession_hints": sup,
            })

    # Full document catalog for the current work branch.
    for item in source_files:
        add_doc(SOURCE_BRANCH, remote_ref(SOURCE_BRANCH), item, "current_source")

    # Branch-specific or changed documents only. This prevents inherited copies
    # from being multiplied into an artificial union-of-branches truth corpus.
    source_path_to_sha = {x["path"]: x["sha"] for x in source_files}
    for branch_row in branch_rows:
        name = branch_row["branch"]
        if not branch_row["knowledge_surface"] or name == SOURCE_BRANCH:
            continue
        ref = remote_ref(name)
        for item in ls_tree(ref):
            if Path(item["path"]).suffix.lower() not in DOC_EXTS:
                continue
            if source_path_to_sha.get(item["path"]) == item["sha"]:
                continue
            add_doc(name, ref, item, "branch_specific")

    def dedupe(rows: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for row in sorted(rows, key=lambda r: (r["branch"], r["path"], r["blob_sha"])):
            key = (row["branch"], row["path"], row["blob_sha"])
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    write_jsonl(OUT / "DOCUMENT_CATALOG.jsonl", dedupe(doc_rows))
    write_jsonl(OUT / "FAILURE_LESSON_CATALOG.jsonl", dedupe(failure_rows))
    write_jsonl(OUT / "FRONTIER_RESEARCH_CATALOG.jsonl", dedupe(research_rows))
    write_jsonl(OUT / "SUPERSESSION_CATALOG.jsonl", dedupe(supersession_rows))

    summary = {
        "schema": "alice-context-catalog-status-v1",
        "source_branch": SOURCE_BRANCH,
        "source_commit": source_sha,
        "branch_count": len(branch_rows),
        "knowledge_branch_count": sum(1 for x in branch_rows if x["knowledge_surface"]),
        "source_file_count": len(source_rows),
        "active_n0_state_present": active_n0_state is not None,
        "document_pointer_count": len(dedupe(doc_rows)),
        "failure_lesson_pointer_count": len(dedupe(failure_rows)),
        "frontier_research_pointer_count": len(dedupe(research_rows)),
        "supersession_pointer_count": len(dedupe(supersession_rows)),
        "authority": "routing-only",
        "notes": [
            "Catalog entries are source pointers, not truth promotions.",
            "Branch-specific records stay branch-qualified.",
            "Private external sources are intentionally excluded from public Git.",
        ],
    }
    write_json(OUT / "CATALOG_STATUS.json", summary)


if __name__ == "__main__":
    main()
