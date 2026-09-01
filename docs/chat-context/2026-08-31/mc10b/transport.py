from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

RUNTIME_ARCHIVE_NAME = "ollama-linux-amd64.tar.zst"
RUNTIME_ARCHIVE_SIZE = 1422416084
RUNTIME_ARCHIVE_SHA256 = "50539C5FE9BF85887733355098DCDB266B433CB8C73FA180713417E9ED6E42BB"
RUNTIME_BINARY_SHA256 = "EB99A47AAD366636488EBD9C163A9180254DFFCFDFE359939F9AABC36E2399C8"
PRIVATE_BLOB_NAME = "mc10b1-private-input.bin"
SOURCE_ROOT_DIR_MARKERS = (
    "mc10a", "activation", "v1", "h11", "router", "semantic", "leak",
    "recon", "mc6", "mc7", "mc8", "mc9", "doctrine", "repo",
)
SOURCE_ROOT_FILE_MARKER = "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json"
FIXED_ZIP_DT = (2026, 1, 1, 0, 0, 0)
ZSTANDARD_PIP_SPEC = "zstandard==0.25.0"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _normalize_member_name(raw: str) -> tuple[str, bool]:
    require(isinstance(raw, str) and raw != "", "empty ZIP member name")
    require("\x00" not in raw, "NUL in ZIP member name")
    is_dir = raw.endswith(("/", "\\"))
    s = raw.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    require(not s.startswith("/"), f"absolute ZIP member: {raw!r}")
    require(not re.match(r"^[A-Za-z]:", s), f"drive-qualified ZIP member: {raw!r}")
    parts = [p for p in s.split("/") if p not in ("", ".")]
    require(parts and all(p != ".." for p in parts), f"unsafe ZIP member: {raw!r}")
    norm = "/".join(parts)
    if is_dir:
        norm += "/"
    return norm, is_dir


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def validate_source_tree(candidate: Path) -> None:
    require(candidate.is_dir(), f"private source root missing: {candidate}")
    require((candidate / SOURCE_ROOT_FILE_MARKER).is_file(), f"private source marker missing: {SOURCE_ROOT_FILE_MARKER}")
    missing = [name for name in SOURCE_ROOT_DIR_MARKERS if not (candidate / name).is_dir()]
    require(not missing, f"private source marker directories missing: {missing}")


def inspect_private_archive(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"private archive missing: {path}")
    normalized: list[tuple[str, bool, zipfile.ZipInfo]] = []
    with zipfile.ZipFile(path, "r") as z:
        require(z.testzip() is None, "private archive CRC failure")
        for info in z.infolist():
            require(not _is_zip_symlink(info), f"symlink ZIP member forbidden: {info.filename}")
            norm, is_dir = _normalize_member_name(info.filename)
            normalized.append((norm, is_dir, info))
    names = [n for n, _, _ in normalized]
    require(len(names) == len(set(names)), "duplicate ZIP members after separator normalization")
    marker_paths = [n for n in names if n == SOURCE_ROOT_FILE_MARKER or n.endswith("/" + SOURCE_ROOT_FILE_MARKER)]
    prefixes: list[str] = []
    for marker_path in marker_paths:
        prefix = marker_path[: -len(SOURCE_ROOT_FILE_MARKER)]
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        dirs_ok = all(any(n.startswith(prefix + d + "/") for n in names) for d in SOURCE_ROOT_DIR_MARKERS)
        file_ok = prefix + SOURCE_ROOT_FILE_MARKER in names
        if dirs_ok and file_ok:
            prefixes.append(prefix)
    prefixes = sorted(set(prefixes))
    require(len(prefixes) == 1, f"expected exactly one structurally valid source prefix in private archive, found {len(prefixes)}")
    prefix = prefixes[0]
    # The canonical private archive contains only the source tree. No hidden sidecar bytes are allowed inside it.
    unexpected = [n for n in names if not n.startswith(prefix)] if prefix else []
    require(not unexpected, f"unexpected files outside canonical source prefix: {unexpected[:10]}")
    return {
        "source_prefix": prefix,
        "member_count": len(names),
        "file_count": sum(1 for _, is_dir, _ in normalized if not is_dir),
        "archive_sha256": sha256_file(path),
    }


def safe_extract_private_archive(path: Path, destination: Path) -> Path:
    info = inspect_private_archive(path)
    require(not destination.exists(), f"private extraction destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    with zipfile.ZipFile(path, "r") as z:
        for member in z.infolist():
            require(not _is_zip_symlink(member), f"symlink ZIP member forbidden: {member.filename}")
            norm, is_dir = _normalize_member_name(member.filename)
            require(norm not in seen, f"duplicate normalized ZIP member: {norm}")
            seen.add(norm)
            rel = PurePosixPath(norm.rstrip("/"))
            target = destination.joinpath(*rel.parts)
            target_resolved_parent = target.parent.resolve()
            dest_resolved = destination.resolve()
            require(target_resolved_parent == dest_resolved or dest_resolved in target_resolved_parent.parents,
                    f"ZIP extraction escaped destination: {member.filename}")
            if is_dir or member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    prefix = str(info["source_prefix"]).strip("/")
    source_root = destination if not prefix else destination.joinpath(*PurePosixPath(prefix).parts)
    validate_source_tree(source_root)
    return source_root


def pack_private_source(source: Path, output: Path, prefix: str = "source") -> dict[str, Any]:
    validate_source_tree(source)
    require(not source.is_symlink(), "source root symlink forbidden")
    prefix = prefix.replace("\\", "/").strip("/")
    require(prefix and ".." not in PurePosixPath(prefix).parts, "invalid archive prefix")
    files = sorted((p for p in source.rglob("*") if p.is_file()), key=lambda p: p.relative_to(source).as_posix())
    require(files, "source tree has no files")
    for p in source.rglob("*"):
        require(not p.is_symlink(), f"source symlink forbidden: {p}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".new")
    tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as z:
        for p in files:
            rel = p.relative_to(source).as_posix()
            arcname = f"{prefix}/{rel}"
            zi = zipfile.ZipInfo(arcname, date_time=FIXED_ZIP_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 3
            zi.external_attr = (0o100644 & 0xFFFF) << 16
            with p.open("rb") as f:
                z.writestr(zi, f.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(tmp, output)
    inspected = inspect_private_archive(output)
    require(inspected["source_prefix"] == prefix + "/", "packed private archive prefix drift")
    # Prove Linux/POSIX-safe extraction locally using the same implementation the Kaggle probe/worker import.
    with tempfile.TemporaryDirectory(prefix="mc10b1-pack-selftest-") as td:
        extracted = safe_extract_private_archive(output, Path(td) / "extract")
        validate_source_tree(extracted)
    return inspected


def locate_exact_file(root: Path, name: str) -> Path:
    matches = sorted(p for p in root.rglob(name) if p.is_file())
    require(len(matches) == 1, f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def locate_runtime_archive(root: Path = Path("/kaggle/input")) -> Path:
    candidates = sorted(p for p in root.rglob(RUNTIME_ARCHIVE_NAME) if p.is_file())
    require(candidates, f"{RUNTIME_ARCHIVE_NAME} not mounted")
    exact = []
    observed = []
    for p in candidates:
        size = p.stat().st_size
        digest = sha256_file(p) if size == RUNTIME_ARCHIVE_SIZE else None
        observed.append((str(p), size, digest))
        if size == RUNTIME_ARCHIVE_SIZE and digest == RUNTIME_ARCHIVE_SHA256:
            exact.append(p)
    require(len(exact) == 1, f"expected exactly one pinned runtime archive; observed={observed[:8]}")
    return exact[0]


def _safe_extract_tar(tar_path: Path, destination: Path) -> None:
    # Ollama's pinned Linux runtime legitimately contains internal CUDA soname
    # symlinks (for example libcublas.so.12 -> libcublas.so.12.x.y). Rejecting
    # every link makes the verified upstream archive impossible to extract.
    # Preserve fail-closed behavior by allowing only links whose normalized
    # targets remain inside the fresh extraction root, and let Python's data
    # extraction filter enforce the same boundary at extraction time.
    with tarfile.open(tar_path, "r:") as tf:
        dest = destination.resolve()
        seen_names: set[str] = set()
        for member in tf.getmembers():
            name = member.name.replace("\\", "/")
            require(not name.startswith("/") and not re.match(r"^[A-Za-z]:", name),
                    f"unsafe runtime tar member: {member.name}")
            pure = PurePosixPath(name)
            parts = [p for p in pure.parts if p not in ("", ".")]
            require(parts and all(p != ".." for p in parts),
                    f"unsafe runtime tar member: {member.name}")
            normalized = "/".join(parts)
            require(normalized not in seen_names, f"duplicate runtime tar member: {member.name}")
            seen_names.add(normalized)
            target = destination.joinpath(*parts).resolve(strict=False)
            require(target == dest or dest in target.parents,
                    f"runtime tar member escaped destination: {member.name}")

            require(member.isfile() or member.isdir() or member.issym() or member.islnk(),
                    f"runtime tar special member forbidden: {member.name}")

            if member.issym() or member.islnk():
                link = member.linkname.replace("\\", "/")
                require(link and not link.startswith("/") and not re.match(r"^[A-Za-z]:", link),
                        f"unsafe runtime tar link target: {member.name} -> {member.linkname}")
                link_pure = PurePosixPath(link)
                if member.issym():
                    combined = pure.parent.joinpath(link_pure)
                else:
                    # POSIX tar hard-link targets are archive-root-relative.
                    combined = link_pure
                stack: list[str] = []
                for part in combined.parts:
                    if part in ("", "."):
                        continue
                    if part == "..":
                        require(stack, f"runtime tar link escaped destination: {member.name} -> {member.linkname}")
                        stack.pop()
                    else:
                        stack.append(part)
                require(stack, f"runtime tar link escaped destination: {member.name} -> {member.linkname}")
                link_target = destination.joinpath(*stack).resolve(strict=False)
                require(link_target == dest or dest in link_target.parents,
                        f"runtime tar link escaped destination: {member.name} -> {member.linkname}")

        data_filter = getattr(tarfile, "data_filter", None)
        require(data_filter is not None, "Python tarfile.data_filter unavailable; refusing runtime extraction")
        tf.extractall(destination, filter=data_filter)


def extract_pinned_runtime(archive: Path, destination: Path) -> Path:
    require(archive.is_file(), "runtime archive missing")
    require(archive.stat().st_size == RUNTIME_ARCHIVE_SIZE, "runtime archive size mismatch")
    require(sha256_file(archive) == RUNTIME_ARCHIVE_SHA256, "runtime archive SHA mismatch")
    require(not destination.exists(), f"runtime destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    proc = subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(destination)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        # CPU and GPU preflight intentionally share the same deterministic fallback; no network install.
        try:
            import zstandard as zstd  # type: ignore
        except Exception:
            # Kaggle notebook images can vary in tar/zstd support. Internet is explicitly enabled
            # for both the CPU preflight and T4 worker, so restore the previously proven fallback
            # using a pinned package version rather than an unbounded latest install.
            pip = subprocess.run(
                [sys.executable, "-m", "pip", "install", ZSTANDARD_PIP_SPEC, "-q",
                 "--disable-pip-version-check", "--no-input"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300,
            )
            require(pip.returncode == 0,
                    f"runtime extraction failed and pinned {ZSTANDARD_PIP_SPEC} install failed; "
                    f"tar={proc.stdout[-1500:]} pip={pip.stdout[-1500:]}")
            import importlib
            importlib.invalidate_caches()
            import zstandard as zstd  # type: ignore
        tar_path = destination.parent / (destination.name + ".runtime.tar")
        try:
            with archive.open("rb") as src, tar_path.open("wb") as dst:
                zstd.ZstdDecompressor().copy_stream(src, dst)
            _safe_extract_tar(tar_path, destination)
        finally:
            tar_path.unlink(missing_ok=True)
    bins = sorted(p for p in destination.rglob("ollama") if p.is_file())
    exact = [p for p in bins if sha256_file(p) == RUNTIME_BINARY_SHA256]
    require(len(exact) == 1, f"expected exactly one pinned Ollama binary after extraction, found {len(exact)}")
    binary = exact[0]
    binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--source", required=True)
    p_pack.add_argument("--output", required=True)
    p_pack.add_argument("--prefix", default="source")
    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--archive", required=True)
    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--archive", required=True)
    p_extract.add_argument("--destination", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "pack":
        result = pack_private_source(Path(args.source), Path(args.output), args.prefix)
    elif args.cmd == "inspect":
        result = inspect_private_archive(Path(args.archive))
    else:
        root = safe_extract_private_archive(Path(args.archive), Path(args.destination))
        result = {"source_root": str(root), "ok": True}
    print(_json_dump(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
