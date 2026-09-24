from __future__ import annotations

import hashlib
import subprocess
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


def require_clean_exact_revision(
    *,
    expected_revision: str,
    label: str,
) -> str:
    root=repository_root()
    expected=str(expected_revision).strip().lower()
    if len(expected)!=40 or any(ch not in "0123456789abcdef" for ch in expected):
        raise SystemExit(f"{label} expected revision must be exact 40-hex")
    status=subprocess.check_output(
        ["git","status","--porcelain"],cwd=root,text=True
    )
    if status.strip():
        raise SystemExit(f"{label} requires a clean exact-source worktree")
    observed=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=root,text=True
    ).strip().lower()
    if observed!=expected:
        raise SystemExit(
            f"{label} source revision drift: observed={observed} expected={expected}"
        )
    return observed
