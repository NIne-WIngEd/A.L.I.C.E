#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    members = archive.getmembers()
    for member in members:
        resolved = (target / member.name).resolve()
        if resolved != root and root not in resolved.parents:
            raise SystemExit(f"unsafe archive member path: {member.name}")
    archive.extractall(target, members=members)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a verified public N0 corpus/tokenizer lineage into Magnolia.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--target-workdir", required=True)
    args = parser.parse_args()

    archive_path = Path(args.archive).resolve()
    receipt_path = Path(args.receipt).resolve()
    target = Path(args.target_workdir).resolve()

    if not archive_path.is_file() or not receipt_path.is_file():
        raise SystemExit("archive and receipt must both exist")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "alice.eipm.n0.external-public-lineage-transfer.v0.1":
        raise SystemExit("unexpected transfer receipt schema")
    for key in ("private_identity_data", "private_identity_gradient", "model_training_performed", "weights_created"):
        if receipt.get(key) is not False:
            raise SystemExit(f"transfer receipt must declare {key}=false")

    actual_archive_sha = sha256_file(archive_path)
    if receipt.get("archive_sha256") != actual_archive_sha:
        raise SystemExit("archive SHA256 does not match transfer receipt")

    if (target / "corpus").exists() or (target / "tokenizer").exists():
        raise SystemExit("target already contains corpus/tokenizer state; refusing overwrite")

    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="n0-lineage-import-", dir=str(target.parent)) as tmp:
        staging = Path(tmp)
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, staging)

        corpus = staging / "corpus"
        tokenizer = staging / "tokenizer"
        manifest_path = staging / "transfer_manifest.json"
        if not corpus.is_dir() or not tokenizer.is_dir() or not manifest_path.is_file():
            raise SystemExit("archive is missing corpus, tokenizer, or transfer_manifest.json")

        if receipt.get("manifest_sha256") != sha256_file(manifest_path):
            raise SystemExit("manifest SHA256 does not match transfer receipt")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "alice.eipm.n0.external-public-lineage-manifest.v0.1":
            raise SystemExit("unexpected transfer manifest schema")
        for key in ("private_identity_data", "private_identity_gradient", "model_training_performed", "weights_created"):
            if manifest.get(key) is not False:
                raise SystemExit(f"transfer manifest must declare {key}=false")

        for rel, expected in manifest.get("files", {}).items():
            path = staging / rel
            if not path.is_file():
                raise SystemExit(f"manifested file missing: {rel}")
            if int(expected.get("bytes", -1)) != path.stat().st_size:
                raise SystemExit(f"size mismatch for {rel}")
            if expected.get("sha256") != sha256_file(path):
                raise SystemExit(f"SHA256 mismatch for {rel}")

        corpus_receipt = corpus / "corpus_receipt.json"
        tokenizer_receipt = tokenizer / "tokenizer_receipt.json"
        tokenizer_json = tokenizer / "tokenizer.json"
        for path in (corpus_receipt, tokenizer_receipt, tokenizer_json):
            if not path.is_file():
                raise SystemExit(f"required lineage artifact missing: {path.name}")

        if receipt.get("corpus_receipt_sha256") != sha256_file(corpus_receipt):
            raise SystemExit("corpus receipt hash mismatch")
        if receipt.get("tokenizer_receipt_sha256") != sha256_file(tokenizer_receipt):
            raise SystemExit("tokenizer receipt hash mismatch")

        tokenizer_meta = json.loads(tokenizer_receipt.read_text(encoding="utf-8"))
        if tokenizer_meta.get("private_identity_data") is not False:
            raise SystemExit("tokenizer receipt must declare private_identity_data=false")
        if tokenizer_meta.get("tokenizer_sha256") != sha256_file(tokenizer_json):
            raise SystemExit("tokenizer hash does not match tokenizer receipt")

        shutil.move(str(corpus), str(target / "corpus"))
        shutil.move(str(tokenizer), str(target / "tokenizer"))
        shutil.copy2(manifest_path, target / "external_transfer_manifest.json")
        shutil.copy2(receipt_path, target / "external_transfer_receipt.json")

    print(json.dumps({
        "status": "PASS",
        "target_workdir": str(target),
        "archive_sha256": actual_archive_sha,
        "git_revision": receipt.get("git_revision"),
        "private_identity_data": False,
        "private_identity_gradient": False,
        "model_training_performed": False,
        "weights_created": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
