"""Same-UID USER+NETWORK isolation and sanitized personality compilation.

Adapted from the audited MFM namespace helper at ae7d680286b84a5755caeb1d2e82e0d09c5176ae.
No MFM corpus, model or teaching objective is used. The P2 stage checks actual
namespace isolation again before resolving or opening any private input.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import ctypes
import errno
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
HOST_UID = 1905
HOST_GID = 100
PACKAGE_SHA256 = "3867ff04d1e326086b9086b2f106b9156b3a3ec8d637d3161e7bf01616183ee9"
MAX_REGISTRY_BYTES = 16 * 1024 * 1024
SUMMARY_SCHEMA = "alice-personality-private-substrate-launch-receipt-v1"


class PrivateStageError(ValueError):
    """A fixed-message privacy/runtime boundary refused the workload."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strip_environment() -> None:
    for name in tuple(os.environ):
        if ("proxy" in name.lower() or name.startswith("BASH_FUNC_")
                or name in {"SSH_AUTH_SOCK", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY",
                            "BASH_ENV", "ENV", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONHOME"}):
            os.environ.pop(name, None)


def _close_inherited_descriptors() -> None:
    for entry in os.listdir("/proc/self/fd"):
        if entry.isdigit() and int(entry) > 2:
            try:
                os.close(int(entry))
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise PrivateStageError("descriptor closure failed") from None


def _write_once(path: str, value: str) -> None:
    fd = os.open(path, os.O_WRONLY)
    try:
        data = value.encode("ascii")
        if os.write(fd, data) != len(data):
            raise PrivateStageError("namespace mapping write was incomplete")
    finally:
        os.close(fd)


def _read_map(path: str) -> list[str]:
    with open(path, encoding="ascii") as stream:
        return stream.read().split()


def _namespace_identity() -> tuple[str, int, int]:
    host_ns = os.environ.get("PERSONALITY_HOST_NETNS", "")
    if re.fullmatch(r"net:\[[0-9]+\]", host_ns) is None:
        raise PrivateStageError("host namespace pin is absent")
    if os.environ.get("PERSONALITY_HOST_UID") != str(HOST_UID) or os.environ.get("PERSONALITY_HOST_GID") != str(HOST_GID):
        raise PrivateStageError("original host identity pin differs")
    return host_ns, HOST_UID, HOST_GID


def _enter() -> dict:
    if sys.platform != "linux":
        raise PrivateStageError("Linux namespace support is required")
    host_ns, uid, gid = _namespace_identity()
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (uid, uid, gid, gid):
        raise PrivateStageError("caller must remain the original nonroot owner")
    if os.readlink("/proc/self/ns/net") != host_ns:
        raise PrivateStageError("outer namespace pin does not match caller")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
    if libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0:
        raise PrivateStageError("fresh USER+NETWORK namespace creation failed")
    _write_once("/proc/self/uid_map", f"{uid} {uid} 1\n")
    _write_once("/proc/self/setgroups", "deny\n")
    _write_once("/proc/self/gid_map", f"{gid} {gid} 1\n")
    if (os.geteuid(), os.getegid()) != (uid, gid):
        raise PrivateStageError("namespace changed the original owner identity")
    if _read_map("/proc/self/uid_map") != [str(uid), str(uid), "1"] or _read_map("/proc/self/gid_map") != [str(gid), str(gid), "1"]:
        raise PrivateStageError("namespace mapping differs from same-UID/GID mapping")
    namespace = os.readlink("/proc/self/ns/net")
    interfaces = sorted(name for _, name in socket.if_nameindex())
    if namespace == host_ns or interfaces != ["lo"]:
        raise PrivateStageError("fresh namespace must differ from host and expose only loopback")
    os.environ["PERSONALITY_ISOLATED_NETNS"] = namespace
    _strip_environment()
    _close_inherited_descriptors()
    return {"host_uid": uid, "host_gid": gid, "host_netns": host_ns,
            "isolated_netns": namespace, "interfaces": interfaces}


def _p2_guard() -> dict:
    """Verify the actual P2 Python process before any private path operation."""
    host_ns, uid, gid = _namespace_identity()
    namespace = os.readlink("/proc/self/ns/net")
    interfaces = sorted(name for _, name in socket.if_nameindex())
    if (namespace == host_ns or namespace != os.environ.get("PERSONALITY_ISOLATED_NETNS")
            or interfaces != ["lo"]):
        raise PrivateStageError("actual container process lacks fresh loopback-only isolation")
    if _read_map("/proc/self/uid_map") != [str(uid), str(uid), "1"] or _read_map("/proc/self/gid_map") != [str(gid), str(gid), "1"]:
        raise PrivateStageError("actual container process lacks same-identity namespace mapping")
    # P2 simulates root inside its container; this is not a root host mapping.
    if os.geteuid() not in (0, uid) or os.getegid() not in (0, gid):
        raise PrivateStageError("container-emulated identity differs from the pinned host route")
    _strip_environment()
    _close_inherited_descriptors()
    return {"host_uid": uid, "host_gid": gid, "host_netns": host_ns,
            "isolated_netns": namespace, "interfaces": interfaces,
            "p2_effective_uid": os.geteuid(), "p2_effective_gid": os.getegid()}


def _bounded_path(value: str, root: Path, *, file: bool) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)):
        raise PrivateStageError("input/output path must be absolute without linked ancestors")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or (file and not resolved.is_file()) or (not file and not resolved.is_dir()):
        raise PrivateStageError("input/output path is outside the owner compute root")
    return resolved


def _file_hash(path: Path, *, max_bytes: int = MAX_REGISTRY_BYTES) -> str:
    digest = sha256()
    before = path.stat()
    if before.st_size > max_bytes:
        raise PrivateStageError("raw lineage exceeds the declared byte bound")
    consumed = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            consumed += len(block)
            if consumed > max_bytes:
                raise PrivateStageError("raw lineage exceeds the declared byte bound")
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise PrivateStageError("input changed during hashing")
    return digest.hexdigest()


def _registry_json(payload: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise PrivateStageError("raw lineage contains duplicate JSON keys")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise PrivateStageError("raw lineage contains a nonfinite JSON constant")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)


def _clean_code(root: Path, expected: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise PrivateStageError("code needs an exact external commit pin")
    base = ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
            f"--git-dir={root / '.git'}", f"--work-tree={root}"]
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0",
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    head = subprocess.run([*base, "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=15, check=True, env=env).stdout.strip()
    dirty = subprocess.run([*base, "status", "--porcelain"], capture_output=True, text=True,
                           timeout=15, check=True, env=env).stdout.strip()
    if head != expected or dirty:
        raise PrivateStageError("public code differs from the exact clean source pin")


def _safe_reason(exc: BaseException) -> str:
    # Explicit fixed source-validation categories; no raw payload/path/traceback.
    reasons = {"raw EINF support is absent from the explicit lineage registry": "missing_raw_inference_lineage",
               "unresolved or wrong-namespace E0 support": "unresolved_source_evidence",
               "unresolved or wrong-namespace EINF support": "unresolved_curated_inference",
               "archive SHA256 differs from the explicit package pin": "source_archive_pin_mismatch"}
    return reasons.get(str(exc), "private_stage_refused")


def _write_summary(path: Path, value: dict) -> None:
    value["receipt_sha256"] = sha256(_canonical(value)).hexdigest()
    with path.open("xb") as stream:
        stream.write(_canonical(value) + b"\n")
    os.chmod(path, 0o600)


def run_private_compile() -> dict:
    """One isolated scientific stage; never a model or gradient operation."""
    isolation = _p2_guard()  # MUST precede private path resolution, stat or hash.
    os.umask(0o077)
    root = Path(os.environ["COMPUTE_ROOT"]).resolve(strict=True)
    code = _bounded_path(os.environ["PERSONALITY_REPO_ROOT"], root, file=False)
    run = _bounded_path(os.environ["PERSONALITY_RUN_ROOT"], root, file=False)
    if stat.S_IMODE(run.stat().st_mode) != 0o700 or run.stat().st_uid not in (0, HOST_UID):
        raise PrivateStageError("fresh output root must be owner-only")
    if run == code or code in run.parents or run in code.parents or any(run.iterdir()):
        raise PrivateStageError("output root must be fresh and separate from public code")
    expected_commit = os.environ["PERSONALITY_EXPECTED_REVISION"]
    _clean_code(code, expected_commit)
    summary_path = run / "compile_summary.json"
    if summary_path.exists() or summary_path.is_symlink():
        raise PrivateStageError("earlier compilation evidence must not be replaced")
    summary = {"schema": SUMMARY_SCHEMA, "state": "FAILED_UNQUALIFIED", "source_commit": expected_commit,
               "isolation": isolation, "private_gradient_authorized": False, "acceptance_authority": False,
               "model_training_performed": False, "weights_created": False, "behavior_qualification": None}
    phase = "public_code_import"
    try:
        sys.path.insert(0, str(code))
        from src.alice_personality.n1.compiler import compile_package, curated_frontier_v2_pin, verify_compiled
        if "torch" in sys.modules or "transformers" in sys.modules:
            raise PrivateStageError("source compiler must not import model runtimes")
        pin = curated_frontier_v2_pin()  # Public pin metadata only.
        if pin.archive_sha256 != PACKAGE_SHA256:
            raise PrivateStageError("public curated frontier pin differs from the external archive identity")
        phase = "private_source_custody"
        package = _bounded_path(os.environ["PRIVATE_PACKAGE_PATH"], root, file=True)
        if code == package or code in package.parents:
            raise PrivateStageError("private source must be separate from the public code tree")
        registry = None
        registry_path = os.environ.get("PERSONALITY_RAW_LINEAGE_PATH", "")
        registry_sha256 = os.environ.get("PERSONALITY_RAW_LINEAGE_SHA256", "")
        if bool(registry_path) != bool(registry_sha256):
            raise PrivateStageError("raw lineage needs both path and external SHA256")
        if registry_path:
            if re.fullmatch(r"[0-9a-f]{64}", registry_sha256) is None:
                raise PrivateStageError("raw lineage external pin is invalid")
            lineage = _bounded_path(registry_path, root, file=True)
            if code == lineage or code in lineage.parents:
                raise PrivateStageError("raw lineage must be separate from public code")
            if lineage.stat().st_size > MAX_REGISTRY_BYTES or _file_hash(lineage, max_bytes=MAX_REGISTRY_BYTES) != registry_sha256:
                raise PrivateStageError("bounded raw lineage differs from external pin")
            with lineage.open("rb") as stream:
                payload = stream.read(MAX_REGISTRY_BYTES + 1)
            if len(payload) > MAX_REGISTRY_BYTES or sha256(payload).hexdigest() != registry_sha256:
                raise PrivateStageError("raw lineage changed before parsing")
            registry = _registry_json(payload)
        phase = "source_compilation"
        output = run / "compiled-substrate"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            receipt = compile_package(package, output, pin=pin, raw_lineage_registry=registry)
            verified = verify_compiled(output, expected_source_archive_sha256=PACKAGE_SHA256,
                                       expected_receipt_sha256=receipt["receipt_sha256"])
        if verified != receipt or receipt["state"] != "COMPILED_UNQUALIFIED" or receipt["private_gradient_authorized"] is not False or receipt["acceptance_authority"] is not False:
            raise PrivateStageError("compiled candidate source cannot grant acceptance or gradients")
        phase = "publication_checks"
        _clean_code(code, expected_commit)
        for item in output.iterdir():
            if item.is_symlink() or not item.is_file():
                raise PrivateStageError("compiled artifact membership is invalid")
            os.chmod(item, 0o600)
        os.chmod(output, 0o700)
        if any(stat.S_IMODE(item.stat().st_mode) != 0o600 for item in output.iterdir()):
            raise PrivateStageError("compiled artifact permissions differ")
        summary.update(state="COMPILED_UNQUALIFIED", source_package_sha256=PACKAGE_SHA256,
                       compiled_receipt_sha256=receipt["receipt_sha256"],
                       active_record_count=receipt["active_record_count"], source_family_count=receipt["source_family_count"],
                       kind_counts=receipt["kind_counts"], split_counts=receipt["split_counts"],
                       files=receipt["files"], raw_lineage_file_sha256=registry_sha256 or None,
                       voice_overlay_ingested=False, unknown_is_behavior_void=False,
                       alternatives_are_unordered_not_negatives=True)
    except BaseException as exc:
        summary.update(failure_phase=phase, failure_type=type(exc).__name__, failure_reason=_safe_reason(exc))
        _write_summary(summary_path, summary)
        raise PrivateStageError("private compilation refused; sanitized failure receipt preserved") from None
    _write_summary(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["--p2-compile-stage"]:
            receipt = run_private_compile()
            print(json.dumps({key: receipt[key] for key in ("schema", "state", "receipt_sha256", "active_record_count",
                             "source_family_count", "private_gradient_authorized", "acceptance_authority")}, sort_keys=True))
            return 0
        if len(args) < 2 or args[:2] != ["--", "/bin/bash"]:
            print("usage: namespace helper -- /bin/bash [args]", file=sys.stderr)
            return 2
        isolation = _enter()
        print(json.dumps({"stage": "namespace", "state": "ISOLATED_UNQUALIFIED", **isolation}, sort_keys=True), flush=True)
        os.execv("/bin/bash", args[1:])
    except BaseException as exc:
        print(json.dumps({"state": "FAILED_UNQUALIFIED", "failure_type": type(exc).__name__,
                          "failure_reason": _safe_reason(exc)}, sort_keys=True), file=sys.stderr)
        return 3
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
