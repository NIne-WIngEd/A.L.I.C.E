#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OWNER = "mkrayanyan"
REPOSITORY = "https://github.com/NIne-WIngEd/A.L.I.C.E.git"
BRANCH = "alice-eipm-v1-build"
# Pin the build content that introduced the governed Kaggle CPU lineage exporter/importer.
EIPM_REVISION = "cb23ac483d0ec5bcc9107f895ada0a6a9c8c34ba"
REMOTE_SCRIPT = "scripts/eipm/n0/kaggle_cpu_real_lineage.sh"
CHARS_PER_SOURCE = 100_000_000
SHARD_MB = 128
POLL_SECONDS = 30
INITIAL_STATUS_DELAY_SECONDS = 30


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_utf8_no_bom(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise RuntimeError(f"UTF-8 BOM detected in {path}")
    json.loads(raw.decode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


REQUEST = {
    "schema": "alice.eipm.n0.kaggle-cpu-lineage-request.v1.0.2",
    "owner": OWNER,
    "repository": REPOSITORY,
    "branch": BRANCH,
    "revision": EIPM_REVISION,
    "remote_script": REMOTE_SCRIPT,
    "chars_per_source": CHARS_PER_SOURCE,
    "shard_mb": SHARD_MB,
    "private_identity_data": False,
    "model_training": False,
    "enable_gpu": False,
    "enable_internet": True,
}
REQUEST_BYTES = json.dumps(REQUEST, sort_keys=True, separators=(",", ":")).encode("utf-8")
REQUEST_SHA256 = sha256_bytes(REQUEST_BYTES)
SLUG = f"rayan-n0-cpu-v102-{REQUEST_SHA256[:12]}"
TITLE = SLUG.replace("-", " ")
KERNEL_REF = f"{OWNER}/{SLUG}"

LOCAL_ROOT = Path.home() / "Downloads" / "RAYAN_N0_KAGGLE_CPU_V102"
STAGE = Path(tempfile.gettempdir()) / f"rayan-n0-kaggle-cpu-v102-{REQUEST_SHA256[:12]}"
OUTPUT_DIR = LOCAL_ROOT / "output"
STATE_PATH = LOCAL_ROOT / "dispatch_state.json"
CLI_LOG_DIR = LOCAL_ROOT / "cli"


RUNNER = f'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = {REPOSITORY!r}
BRANCH = {BRANCH!r}
REVISION = {EIPM_REVISION!r}
REMOTE_SCRIPT = {REMOTE_SCRIPT!r}
CHARS_PER_SOURCE = {CHARS_PER_SOURCE}
SHARD_MB = {SHARD_MB}

scratch = Path('/tmp/rayan-n0-lineage-v102')
repo = scratch / 'repo'
workdir = scratch / 'n0-v01'
hf_home = scratch / 'hf-cache'
export_dir = Path('/kaggle/working/rayan-n0-export')
export_dir.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_receipt(status: str, **extra) -> None:
    payload = {{
        'schema': 'alice.eipm.n0.kaggle-cpu-runtime.v1.0.2',
        'status': status,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_revision': REVISION,
        'private_identity_data': False,
        'private_identity_gradient': False,
        'model_training_performed': False,
        'weights_created': False,
        **extra,
    }}
    (export_dir / 'kaggle_runtime_receipt.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8'
    )

try:
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    print('===== RAYAN N0 KAGGLE CPU RUNTIME =====', flush=True)
    print('revision=' + REVISION, flush=True)
    print('accelerator=none', flush=True)
    print('internet=true', flush=True)
    print('private_identity_data=false', flush=True)
    print('model_training=false', flush=True)

    subprocess.run([
        'git', 'clone', '--depth', '50', '--branch', BRANCH, REPO_URL, str(repo)
    ], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f'git revision mismatch: expected={{REVISION}} actual={{actual}}')

    env = os.environ.copy()
    env['ALICE_N0_WORKDIR'] = str(workdir)
    env['N0_KAGGLE_EXPORT_DIR'] = str(export_dir)
    env['HF_HOME'] = str(hf_home)
    env['N0_BOOTSTRAP_CHARS_PER_SOURCE'] = str(CHARS_PER_SOURCE)
    env['N0_SHARD_MB'] = str(SHARD_MB)
    env['TOKENIZERS_PARALLELISM'] = 'true'

    subprocess.run(['bash', str(repo / REMOTE_SCRIPT)], cwd=str(repo), env=env, check=True)

    archive = export_dir / 'rayan-n0-v01-public-lineage.tar.gz'
    transfer_receipt = export_dir / 'transfer_receipt.json'
    manifest = export_dir / 'transfer_manifest.json'
    for required in (archive, transfer_receipt, manifest):
        if not required.is_file():
            raise RuntimeError(f'missing required export: {{required}}')

    receipt = json.loads(transfer_receipt.read_text(encoding='utf-8'))
    actual_archive_hash = sha256_file(archive)
    if actual_archive_hash != receipt.get('archive_sha256'):
        raise RuntimeError('archive SHA-256 does not match transfer receipt')
    if receipt.get('git_revision') != REVISION:
        raise RuntimeError('transfer receipt git revision mismatch')

    write_receipt(
        'PASS',
        archive_sha256=actual_archive_hash,
        archive_bytes=archive.stat().st_size,
        transfer_receipt_sha256=sha256_file(transfer_receipt),
        transfer_manifest_sha256=sha256_file(manifest),
    )
    print('RAYAN_N0_KAGGLE_CPU_RUNTIME=PASS', flush=True)
except BaseException as exc:
    write_receipt(
        'FAIL',
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=traceback.format_exc(),
    )
    print('RAYAN_N0_KAGGLE_CPU_RUNTIME=FAIL', flush=True)
    traceback.print_exc()
    raise
finally:
    shutil.rmtree(scratch, ignore_errors=True)
'''


METADATA = {
    "id": KERNEL_REF,
    "title": TITLE,
    "code_file": "runner.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "machine_shape": "",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}


def slug_from_title(title: str) -> str:
    return "-".join(title.lower().split())


def stage_and_validate() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for child in STAGE.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    runner_path = STAGE / "runner.py"
    runner_path.write_text(RUNNER, encoding="utf-8", newline="\n")
    metadata_path = STAGE / "kernel-metadata.json"
    write_json_utf8_no_bom(metadata_path, METADATA)

    compile(runner_path.read_text(encoding="utf-8"), str(runner_path), "exec")
    parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_slug = slug_from_title(parsed["title"])
    actual_slug = parsed["id"].split("/", 1)[1]
    if expected_slug != actual_slug:
        raise RuntimeError(f"title/slug binding failed: {expected_slug} != {actual_slug}")

    staged_names = sorted(p.name for p in STAGE.iterdir() if p.is_file())
    if staged_names != ["kernel-metadata.json", "runner.py"]:
        raise RuntimeError(f"single-code-file staging contract failed: {staged_names}")

    print("python_compile_gate_passed=true files=1")
    print("metadata_utf8_no_bom=true")
    print("metadata_json_parse=true")
    print("title_slug_binding=true")
    print("single_code_file_transport=true")
    print("deterministic_kernel_identity=true")


def find_kaggle() -> str:
    for candidate in ("kaggle.exe", "kaggle"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("Kaggle CLI not found on PATH")


def run_cli(kaggle: str, args: list[str], label: str) -> subprocess.CompletedProcess[str]:
    CLI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    save_text(CLI_LOG_DIR / f"{label}.stdout.txt", proc.stdout)
    save_text(CLI_LOG_DIR / f"{label}.stderr.txt", proc.stderr)
    write_json_utf8_no_bom(
        CLI_LOG_DIR / f"{label}.receipt.json",
        {
            "argv": cmd,
            "exit_code": proc.returncode,
            "timestamp_utc": now_iso(),
            "stdout_sha256": sha256_bytes(proc.stdout.encode("utf-8", errors="replace")),
            "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8", errors="replace")),
        },
    )
    return proc


def terminal_status(text: str) -> str | None:
    upper = text.upper()
    if "COMPLETE" in upper:
        return "COMPLETE"
    if "ERROR" in upper or "FAILED" in upper or "FAILURE" in upper:
        return "ERROR"
    if "CANCELLED" in upper or "CANCELED" in upper:
        return "CANCELLED"
    if "RUNNING" in upper:
        return "RUNNING"
    if "QUEUED" in upper:
        return "QUEUED"
    return None


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {
            "schema": "alice.eipm.n0.kaggle-cpu-dispatch-state.v1.0.2",
            "request_sha256": REQUEST_SHA256,
            "kernel_ref": KERNEL_REF,
            "push_attempted": False,
            "push_acknowledged": False,
            "output_verified": False,
            "created_at": now_iso(),
        }
    state = read_json(STATE_PATH)
    if state.get("request_sha256") != REQUEST_SHA256 or state.get("kernel_ref") != KERNEL_REF:
        raise RuntimeError("Existing dispatch state does not match this deterministic request")
    return state


def persist_state(state: dict[str, Any]) -> None:
    write_json_utf8_no_bom(STATE_PATH, state)


def status_once(kaggle: str, seq: int) -> tuple[str | None, subprocess.CompletedProcess[str]]:
    proc = run_cli(kaggle, ["kernels", "status", KERNEL_REF], f"status-{seq:05d}")
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    return terminal_status(combined), proc


def retrieve_output(kaggle: str, label: str) -> subprocess.CompletedProcess[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return run_cli(kaggle, ["kernels", "output", KERNEL_REF, "-p", str(OUTPUT_DIR), "-o"], label)


def find_unique(name: str) -> Path:
    matches = [p for p in OUTPUT_DIR.rglob(name) if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(matches)}: {matches}")
    return matches[0]


def verify_downloaded_output() -> dict[str, Any]:
    archive = find_unique("rayan-n0-v01-public-lineage.tar.gz")
    receipt_path = find_unique("transfer_receipt.json")
    manifest = find_unique("transfer_manifest.json")
    runtime_receipt_path = find_unique("kaggle_runtime_receipt.json")

    receipt = read_json(receipt_path)
    runtime_receipt = read_json(runtime_receipt_path)
    if runtime_receipt.get("status") != "PASS":
        raise RuntimeError(f"remote runtime receipt is not PASS: {runtime_receipt}")
    if receipt.get("git_revision") != EIPM_REVISION:
        raise RuntimeError("downloaded transfer receipt revision mismatch")
    if receipt.get("private_identity_data") is not False:
        raise RuntimeError("unexpected private identity data flag")
    if receipt.get("model_training_performed") is not False:
        raise RuntimeError("unexpected model training flag")

    actual_archive_hash = sha256_file(archive)
    if actual_archive_hash != receipt.get("archive_sha256"):
        raise RuntimeError(
            f"archive hash mismatch expected={receipt.get('archive_sha256')} actual={actual_archive_hash}"
        )

    return {
        "archive": str(archive),
        "archive_sha256": actual_archive_hash,
        "archive_bytes": archive.stat().st_size,
        "transfer_receipt": str(receipt_path),
        "transfer_receipt_sha256": sha256_file(receipt_path),
        "transfer_manifest": str(manifest),
        "transfer_manifest_sha256": sha256_file(manifest),
        "runtime_receipt": str(runtime_receipt_path),
        "git_revision": receipt.get("git_revision"),
    }


def best_effort_cleanup(kaggle: str) -> None:
    proc = run_cli(kaggle, ["kernels", "delete", KERNEL_REF, "-y"], "cleanup-kernel")
    if proc.returncode == 0:
        print("remote_cleanup_kernel=OK")
    else:
        print("remote_cleanup_kernel=WARNING_PRESERVED")
        if proc.stderr.strip():
            print(proc.stderr.strip())


def main() -> int:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    stage_and_validate()
    kaggle = find_kaggle()

    print("===== RAYAN N0 KAGGLE CPU v1.0.2 =====")
    print(f"PYTHON={sys.executable}")
    print(f"request_sha256={REQUEST_SHA256}")
    print(f"kernel_ref={KERNEL_REF}")
    print(f"stage={STAGE}")
    print(f"output_dir={OUTPUT_DIR}")
    print("dispatch=official_kaggle_cli_direct")
    print("accelerator=none")
    print("internet=true")
    print("private_kernel=true")
    print("push_once_contract=true")
    print("resume_same_identity=true")
    print("native_exit_receipts_preserved=true")
    print("private_identity_data=false")
    print("model_training=false")

    version = run_cli(kaggle, ["--version"], "kaggle-version")
    if version.returncode != 0:
        print(version.stdout)
        print(version.stderr, file=sys.stderr)
        return 2
    print(version.stdout.strip() or version.stderr.strip())

    state = load_state()
    persist_state(state)

    # Reconcile the deterministic identity before deciding whether a push is allowed.
    status_seq = int(state.get("status_seq", 0)) + 1
    status, status_proc = status_once(kaggle, status_seq)
    state["status_seq"] = status_seq
    persist_state(state)

    remote_visible = status_proc.returncode == 0 and status is not None
    if remote_visible:
        state["push_acknowledged"] = True
        state["last_status"] = status
        persist_state(state)
        print(f"reconciled_existing_kernel=true status={status}")
    elif not state.get("push_attempted", False):
        state["push_attempted"] = True
        state["push_intent_at"] = now_iso()
        persist_state(state)
        print("phase=KERNEL_PUSH")
        push = run_cli(kaggle, ["kernels", "push", "-p", str(STAGE)], "push")
        state["push_exit_code"] = push.returncode
        state["push_finished_at"] = now_iso()
        state["push_stdout_sha256"] = sha256_bytes(push.stdout.encode("utf-8", errors="replace"))
        state["push_stderr_sha256"] = sha256_bytes(push.stderr.encode("utf-8", errors="replace"))
        if push.returncode != 0:
            persist_state(state)
            print(push.stdout)
            print(push.stderr, file=sys.stderr)
            print("push_once_guard=preserved_no_automatic_repush")
            return 3
        state["push_acknowledged"] = True
        persist_state(state)
        print(push.stdout.strip())
        print(f"initial_status_delay_seconds={INITIAL_STATUS_DELAY_SECONDS}")
        time.sleep(INITIAL_STATUS_DELAY_SECONDS)
    else:
        print("push_intent_already_persisted=true")
        print("automatic_repush=false")
        print("reconciling_exact_identity=true")

    consecutive_unknown = 0
    while True:
        status_seq = int(state.get("status_seq", 0)) + 1
        status, proc = status_once(kaggle, status_seq)
        state["status_seq"] = status_seq
        state["last_status_check_at"] = now_iso()
        if proc.returncode == 0 and status is not None:
            consecutive_unknown = 0
            state["last_status"] = status
            state["push_acknowledged"] = True
            persist_state(state)
            print(f"[{now_iso()}] status={status}")
            if status in {"COMPLETE", "ERROR", "CANCELLED"}:
                break
        else:
            consecutive_unknown += 1
            state["consecutive_unknown_status"] = consecutive_unknown
            persist_state(state)
            combined = (proc.stdout + "\n" + proc.stderr).strip().replace("\n", " | ")
            print(f"[{now_iso()}] status=UNRESOLVED attempt={consecutive_unknown} detail={combined[:500]}")
            # Delayed Kaggle discoverability is not permission to create a second identity.
            if not state.get("push_acknowledged") and consecutive_unknown >= 20:
                print("exact_identity_not_visible_after_persisted_push_intent=true")
                print("automatic_repush=false")
                return 4
        time.sleep(POLL_SECONDS)

    final_status = state.get("last_status")
    print(f"terminal_status={final_status}")
    output_proc = retrieve_output(kaggle, "output")
    if output_proc.returncode != 0:
        print(output_proc.stdout)
        print(output_proc.stderr, file=sys.stderr)
        print("retrieval_failure_never_causes_inference_rerun=true")
        return 5

    if final_status != "COMPLETE":
        print("remote_run_not_complete=true")
        print(f"preserved_output_dir={OUTPUT_DIR}")
        print("remote_kernel_preserved_for_diagnosis=true")
        return 6

    verified = verify_downloaded_output()
    state["output_verified"] = True
    state["verified_output"] = verified
    state["verified_at"] = now_iso()
    persist_state(state)

    print("terminal_output_retrieval=true")
    print("download_hash_validation=true")
    print(json.dumps(verified, indent=2, sort_keys=True))

    best_effort_cleanup(kaggle)
    print("===== RAYAN N0 KAGGLE CPU COMPLETE =====")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nLOCAL_INTERRUPT=true")
        print(f"resume_command=rerun controller; deterministic kernel identity remains {KERNEL_REF}")
        raise SystemExit(130)
