#!/usr/bin/env python3
"""Manifest and verify the exact public N0 P43 input closure; transfer no bytes.

Run beside n0_5e29_paid_portability_input_inventory_v1.py. The manifest uses
logical repo/work roots. Its absolute teacher dependency is checked in place
by the inventory on both hosts. Neither mode touches the sealed FINAL rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

INVENTORY = Path(__file__).with_name("n0_5e29_paid_portability_input_inventory_v1.py")
SCHEMA = "alice.n0.p43.paid-public-input-transfer-manifest.v1"
REVISION = "5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(repo: Path, work: Path, mixture: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(INVENTORY), "--repo", str(repo), "--work", str(work),
         "--mixture", str(mixture)], capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


def relative_entry(path: Path, repo: Path, work: Path) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    for label, root in (("repo", repo), ("work", work)):
        if resolved.is_relative_to(root):
            return label, str(resolved.relative_to(root))
    raise SystemExit(f"STOP: input outside source/work roots: {path}")


def collect(repo: Path, work: Path, mixture: Path, observed: dict) -> list[dict]:
    paths: set[Path] = set()

    def add(record: dict) -> None:
        paths.add(Path(record["path"]))

    for key in ("checkpoint", "corpus_receipt", "tokenizer_json", "teacher_registry",
                "teacher_audit", "mixture_manifest", "mixture_audit", "final_freeze_receipt"):
        add(observed[key])
    tokenizer_json = Path(observed["tokenizer_json"]["path"])
    tokenizer_receipt = tokenizer_json.with_name("tokenizer_receipt.json")
    tokenizer_value = json.loads(tokenizer_receipt.read_text(encoding="utf-8"))
    if (tokenizer_value.get("tokenizer_sha256") != observed["tokenizer_json"]["sha256"]
            or tokenizer_value.get("model_id") != "alice-n0-semantic-v0.2"
            or int(tokenizer_value.get("vocab_size_observed", 0)) != 48000
            or any(tokenizer_value.get(k) is not False for k in
                   ("private_identity_data", "private_identity_gradient", "model_training_performed"))):
        raise SystemExit("STOP: tokenizer receipt does not bind public tokenizer")
    paths.add(tokenizer_receipt)
    for record in observed["exact_head_cpu_receipts"].values():
        add(record)
    for record in observed["optimizer_facing_lanes"].values():
        add(record)
    for pair in observed["teacher_shards"]:
        add(pair["curriculum"])
        add(pair["manifest"])

    receipt = json.loads(Path(observed["corpus_receipt"]["path"]).read_text(encoding="utf-8"))
    corpus = Path(observed["corpus_receipt"]["path"]).parent.resolve()
    for source in receipt["sources"]:
        for shard in source["shards"]:
            rel = Path(shard["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise SystemExit("STOP: unsafe corpus shard path")
            path = (corpus / rel).resolve(strict=True)
            if not path.is_relative_to(corpus):
                raise SystemExit("STOP: corpus shard escaped root")
            paths.add(path)

    # Include the provenance files beside the seven row inputs, as well as the
    # CPU proof copies. Exclude the sealed FINAL subtree except its freeze receipt.
    mixture_root = mixture.resolve(strict=True)
    for path in mixture_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(mixture_root)
        if relative.parts[0] == "final-v2" and relative != Path("final-v2/freeze_receipt.json"):
            continue
        if not path.resolve(strict=True).is_relative_to(mixture_root):
            raise SystemExit(f"STOP: mixture symlink escapes root: {path}")
        paths.add(path)

    entries = []
    for path in paths:
        root, rel = relative_entry(path, repo, work)
        if root == "work" and "final-v2" in Path(rel).parts and not rel.endswith(
            "final-v2/freeze_receipt.json"
        ):
            raise SystemExit("STOP: sealed FINAL input in transfer list")
        entries.append({"root": root, "path": rel, "bytes": path.stat().st_size,
                        "sha256": sha256(path)})
    return sorted(entries, key=lambda e: (e["root"], e["path"]))


def verify_entries(entries: list[dict], repo: Path, work: Path) -> None:
    seen = set()
    if not entries:
        raise SystemExit("STOP: empty transfer manifest")
    for entry in entries:
        label, name = entry["root"], entry["path"]
        if label not in ("repo", "work") or not isinstance(name, str):
            raise SystemExit("STOP: invalid transfer root/path")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or not name or (label, name) in seen:
            raise SystemExit("STOP: unsafe or duplicate transfer path")
        if label == "work" and "final-v2" in rel.parts and not name.endswith(
            "final-v2/freeze_receipt.json"
        ):
            raise SystemExit("STOP: sealed FINAL input in transfer manifest")
        seen.add((label, name))
        root = repo if label == "repo" else work
        path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit(f"STOP: missing/escaping input: {label}/{name}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise SystemExit(f"STOP: transferred input changed: {label}/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    repo, work, mixture = (path.resolve(strict=True) for path in (args.repo, args.work, args.mixture))
    observed = inventory(repo, work, mixture)
    if observed["source_revision"] != REVISION or observed["final_files_staged"] is not False:
        raise SystemExit("STOP: source or FINAL boundary drift")
    if args.mode == "create":
        if args.manifest.exists():
            raise SystemExit("STOP: preserve existing transfer manifest")
        entries = collect(repo, work, mixture, observed)
        verify_entries(entries, repo, work)
        payload = {"schema": SCHEMA, "source_revision": REVISION,
                   "mixture_manifest_sha256": observed["mixture_manifest"]["sha256"],
                   "teacher_registry_sha256": observed["teacher_registry"]["sha256"],
                   "private_identity_data": False, "final_rows_staged": False,
                   "training_authorized": False, "files": entries}
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"CREATED_PUBLIC_INPUT_MANIFEST files={len(entries)} sha256={sha256(args.manifest)}")
        return
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (payload.get("schema") != SCHEMA or payload.get("source_revision") != REVISION
            or payload.get("mixture_manifest_sha256") != observed["mixture_manifest"]["sha256"]
            or payload.get("teacher_registry_sha256") != observed["teacher_registry"]["sha256"]
            or payload.get("private_identity_data") is not False
            or payload.get("final_rows_staged") is not False
            or payload.get("training_authorized") is not False):
        raise SystemExit("STOP: transfer manifest authority/binding drift")
    verify_entries(payload["files"], repo, work)
    if payload["files"] != collect(repo, work, mixture, observed):
        raise SystemExit("STOP: transferred public input closure differs from manifest")
    print(f"VERIFIED_PUBLIC_INPUT_CLOSURE files={len(payload['files'])} sha256={sha256(args.manifest)}")


if __name__ == "__main__":
    main()
