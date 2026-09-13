#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from alice_personality.n1.compiler import compile_identity_substrate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_internal_hashes(root: Path) -> dict[str, int]:
    sums = root / "SHA256SUMS.txt"
    if not sums.exists():
        raise ValueError("package is missing SHA256SUMS.txt")
    verified = 0
    for line_no, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"invalid SHA256SUMS line {line_no}")
        expected, relative = parts
        path = root / relative.strip()
        if not path.is_file():
            raise ValueError(f"manifested file is missing: {relative}")
        if sha256_file(path).lower() != expected.lower():
            raise ValueError(f"SHA256 mismatch for {relative}")
        verified += 1
    return {"verified_files": verified}


def find_package_root(extracted: Path) -> Path:
    if (extracted / "curation_manifest.json").is_file():
        return extracted
    candidates = [path.parent for path in extracted.rglob("curation_manifest.json")]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one curated package root, found {len(candidates)}")
    return candidates[0]


def compile_package(package: Path, output: Path) -> dict:
    if package.is_dir():
        root = find_package_root(package)
        verification = verify_internal_hashes(root)
        receipt = compile_identity_substrate(root, output)
        package_digest = None
    else:
        package_digest = sha256_file(package)
        with zipfile.ZipFile(package) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failure at {bad}")
            with tempfile.TemporaryDirectory(prefix="alice-eipm-n1-") as tmp:
                extracted = Path(tmp)
                archive.extractall(extracted)
                root = find_package_root(extracted)
                verification = verify_internal_hashes(root)
                receipt = compile_identity_substrate(root, output)

    outer = {
        "schema": "alice.eipm.n1.package-compile-receipt.v0.1",
        "source_package_sha256": package_digest,
        "internal_hash_verification": verification,
        "private_gradient_authorized": False,
        "compiled": receipt,
    }
    (output / "package_compile_receipt.json").write_text(
        json.dumps(outer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the private curated EIPM frontier into N1 substrate artifacts."
    )
    parser.add_argument("--package", required=True, help="Curated frontier ZIP or extracted directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    receipt = compile_package(Path(args.package), Path(args.output_dir))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
