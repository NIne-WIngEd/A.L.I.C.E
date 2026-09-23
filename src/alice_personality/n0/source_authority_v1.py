from __future__ import annotations

import hashlib
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_path(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_canonical_source_file(
    supplied: str | Path,
    canonical_relative: str,
    *,
    label: str,
) -> Path:
    supplied_path=Path(supplied).resolve()
    canonical=(repository_root()/canonical_relative).resolve()
    if not supplied_path.is_file():
        raise SystemExit(f"{label} missing")
    if not canonical.is_file():
        raise SystemExit(f"canonical {label} missing from exact source")
    if sha256_path(supplied_path)!=sha256_path(canonical):
        raise SystemExit(f"{label} drift from canonical exact-source file")
    return canonical
