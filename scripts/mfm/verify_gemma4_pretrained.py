"""Seal the exact pretrained (non -it) Gemma 4 12B publisher snapshot.

This script deliberately does not import a model, execute repository code, or
contact the network.  It verifies every byte against the independently pinned
publisher file manifest before any downstream clone or tensor edit is made.
The receipt must live outside the source snapshot and is not a substitute for
the pinned constants in this file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


REPO = "google/gemma-4-12B"
REVISION = "023679ed352de9bb66cc873c9009ce3482585c08"
SCHEMA = "mfm-gemma4-pretrained-source-v1"
FILES = {
    ".gitattributes": (1624, "484fac0cb8b057eefe1992c8b72ac6e7438c7d17bd60c0e278b401c2190f7e72"),
    "README.md": (28340, "131e26f7f1fa69445dc4b0ab98a2251811f8c3128426bf09fcef6ac5ee16e7e4"),
    "config.json": (4383, "14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098"),
    "generation_config.json": (233, "02b56bd11e1cd1e363e701a85a2fd7fbaa2992ec3358c1cd7cc44ead7208f505"),
    "model.safetensors": (23919549408, "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a"),
    "processor_config.json": (1382, "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9"),
    "tokenizer.json": (32170070, "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6"),
    "tokenizer_config.json": (888, "522a38334973725dba8f7c645195b19dda0c284f403f43273f77837679ba2eab"),
}


class SnapshotError(ValueError):
    """The source snapshot is incomplete or differs from the pinned publisher."""


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify(snapshot: Path) -> dict:
    snapshot = snapshot.expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise SnapshotError("snapshot is not a directory")
    actual = {}
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot).as_posix()
        if path.is_symlink() or not path.is_file():
            raise SnapshotError(f"nonregular snapshot entry: {relative}")
        actual[relative] = path
    if set(actual) != set(FILES):
        raise SnapshotError(
            f"snapshot tree mismatch: missing={sorted(set(FILES)-set(actual))}, "
            f"unexpected={sorted(set(actual)-set(FILES))}")
    rows = []
    for name, (size, expected) in FILES.items():
        path = actual[name]
        if path.stat().st_size != size:
            raise SnapshotError(f"wrong size: {name}")
        found = _digest(path)
        if path.stat().st_size != size or found != expected:
            raise SnapshotError(f"wrong publisher content: {name}")
        rows.append({"path": name, "size": size, "sha256": found})
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"]:
        raise SnapshotError("unexpected architecture")
    receipt = {
        "schema": SCHEMA, "repository": REPO, "revision": REVISION,
        "snapshot_path": str(snapshot), "files": rows,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "meaning": "Exact pretrained source bytes verified; no behavioral qualification",
    }
    receipt["receipt_sha256"] = sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("receipt already exists; preserve prior receipt")
    if args.receipt.resolve() == args.snapshot.resolve() or args.snapshot.resolve() in args.receipt.resolve().parents:
        parser.error("receipt must be outside the source snapshot")
    receipt = verify(args.snapshot)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({k: receipt[k] for k in ("repository", "revision", "receipt_sha256", "meaning")}))


if __name__ == "__main__":
    main()
