"""Retained source inspection through the existing original-owner USER+NET/P2."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import sys


HELPER = Path(__file__).with_name("enter_private_network_namespace.py")
SPEC = importlib.util.spec_from_file_location("retained_existing_namespace", HELPER)
namespace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(namespace)  # PUBLIC stdlib guard only; no private I/O.
FLAGS = ("acceptance_authority", "private_gradient_authorized", "historical_authority_granted",
         "training_authorized", "source_admission", "original_source_independently_verified")


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) != 0o600:
            raise namespace.PrivateStageError("output permissions differ")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_receipt(value: dict, inspection: dict, payload: bytes) -> None:
    keys = {"schema", "state", "inspection", "code_commit", "entry_implementation_sha256",
            "isolation", "protected_file_sha256", "receipt_sha256"}
    valid = (type(value) is dict and set(value) == keys and
        value["schema"] == "alice-personality-retained-inference-launch-v1" and
        value["state"] == "RETAINED_SOURCE_INSPECTED_UNQUALIFIED" and
        namespace._canonical(value["inspection"]) == namespace._canonical(inspection) and
        type(value["code_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", value["code_commit"]) is not None and
        type(value["entry_implementation_sha256"]) is dict and
        set(value["entry_implementation_sha256"]) == {"entry.py", "namespace.py"})
    if valid:
        valid = all(type(item) is str and re.fullmatch(r"[0-9a-f]{64}", item) is not None for item in
            [value["protected_file_sha256"], value["receipt_sha256"], *value["entry_implementation_sha256"].values()])
    if valid:
        identity = value["isolation"]
        valid = (type(identity) is dict and set(identity) == {"host_uid", "host_gid", "host_netns", "isolated_netns",
                 "interfaces", "p2_effective_uid", "p2_effective_gid"} and
            type(identity["host_uid"]) is int and identity["host_uid"] == namespace.HOST_UID and
            type(identity["host_gid"]) is int and identity["host_gid"] == namespace.HOST_GID and
            type(identity["p2_effective_uid"]) is int and identity["p2_effective_uid"] in (0, namespace.HOST_UID) and
            type(identity["p2_effective_gid"]) is int and identity["p2_effective_gid"] in (0, namespace.HOST_GID) and
            identity["interfaces"] == ["lo"] and
            all(type(identity[key]) is str and re.fullmatch(r"net:\[[0-9]+\]", identity[key]) is not None
                for key in ("host_netns", "isolated_netns")) and identity["host_netns"] != identity["isolated_netns"])
    if not valid or value["protected_file_sha256"] != sha256(payload).hexdigest() or \
            value["receipt_sha256"] != sha256(namespace._canonical(
                {key: item for key, item in value.items() if key != "receipt_sha256"})).hexdigest():
        raise namespace.PrivateStageError("fixed output binding differs")


def run_inspection() -> dict:
    isolation = namespace._p2_guard()  # FIRST operation, before private env/path work.
    os.umask(0o077)
    root_value = Path(os.environ["COMPUTE_ROOT"])
    if not root_value.is_absolute() or ".." in root_value.parts or any(
            item.is_symlink() or getattr(item.lstat(), "st_file_attributes", 0) & 0x400
            for item in (root_value, *root_value.parents)):
        raise namespace.PrivateStageError("owner boundary differs")
    root = root_value.resolve(strict=True)
    if not root.is_dir() or root.stat().st_uid not in (0, namespace.HOST_UID) or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise namespace.PrivateStageError("owner boundary differs")
    code = namespace._bounded_path(os.environ["PERSONALITY_REPO_ROOT"], root, file=False)
    run = namespace._bounded_path(os.environ["PERSONALITY_RUN_ROOT"], root, file=False)
    if run == code or code in run.parents or run in code.parents or \
            run.stat().st_uid not in (0, namespace.HOST_UID) or stat.S_IMODE(run.stat().st_mode) != 0o700 or any(run.iterdir()):
        raise namespace.PrivateStageError("output requires a fresh separate owner directory")
    revision = os.environ["PERSONALITY_EXPECTED_REVISION"]
    namespace._clean_code(code, revision)
    if namespace._bounded_path(str(Path(__file__).absolute()), root, file=True) != \
            code / "scripts/eipm/n1/inspect_retained_inference_source.py" or \
            namespace._bounded_path(str(HELPER.absolute()), root, file=True) != \
            code / "scripts/eipm/n1/enter_private_network_namespace.py":
        raise namespace.PrivateStageError("public entry location differs")
    entry_code = {"entry.py": sha256(Path(__file__).read_bytes()).hexdigest(),
                  "namespace.py": sha256(HELPER.read_bytes()).hexdigest()}
    sys.path.insert(0, str(code))
    from src.alice_personality.n1 import compiler as c
    from src.alice_personality.n1 import retained_inference_source as retained
    from src.alice_personality.n1 import raw_inference_lineage as raw
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile, LEGACY_RESERVE_SHA256
    pin = c.curated_frontier_v2_pin()
    if pin.archive_sha256 != namespace.PACKAGE_SHA256 or \
            pin.members_sha256.get(pin.package_root + "/" + retained.MEMBER) != LEGACY_RESERVE_SHA256:
        raise namespace.PrivateStageError("reviewed public source pin differs")
    package = namespace._bounded_path(os.environ["PRIVATE_PACKAGE_PATH"], root, file=True)
    if code in package.parents or run in package.parents:
        raise namespace.PrivateStageError("source overlaps public code or new outputs")
    # Keep the exact open source under custody through publication, not just parsing.
    with _PinnedFile(package, pin.archive_sha256, raw.MAX_ARCHIVE_BYTES) as source:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            protected, summary = retained.inspect_retained_source(source.path, pin=pin)
            retained.verify_result(protected, summary)
            if summary["retained_candidate_coverage"] is not None:
                namespace._lineage_coverage(summary["retained_candidate_coverage"], protected["candidate_registry"])
        payload = c._canonical(protected) + b"\n"
        receipt = {"schema": "alice-personality-retained-inference-launch-v1", "state": retained.STATE,
            "inspection": summary, "code_commit": revision, "entry_implementation_sha256": entry_code,
            "isolation": isolation, "protected_file_sha256": sha256(payload).hexdigest()}
        receipt["receipt_sha256"] = sha256(namespace._canonical(receipt)).hexdigest()
        _validate_receipt(receipt, summary, payload)
        source.verify()
        namespace._clean_code(code, revision)
        if retained.implementation_hashes() != summary["implementation_sha256"] or entry_code != {
                "entry.py": sha256(Path(__file__).read_bytes()).hexdigest(), "namespace.py": sha256(HELPER.read_bytes()).hexdigest()}:
            raise namespace.PrivateStageError("executing code changed")
        _write_new(run / "retained-source.protected.json", payload)
        source.verify()
        _write_new(run / "retained-source-summary.json", namespace._canonical(receipt) + b"\n")
    return receipt


def main(argv=None) -> int:
    if list(sys.argv[1:] if argv is None else argv) != ["--p2-retained-source-stage"]:
        print("usage: guarded retained source inspection --p2-retained-source-stage", file=sys.stderr)
        return 2
    try:
        receipt = run_inspection()
        # run_inspection has validated every public field; no protected result is logged.
        print(json.dumps({"state": receipt["state"], "receipt_sha256": receipt["receipt_sha256"],
            **{key: receipt["inspection"][key] for key in ("row_count", "observed_proposal_count", "namespace_contract", *FLAGS)}},
            sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({"state": "FAILED_UNQUALIFIED", "failure_reason": "retained_source_refused",
                          **{key: False for key in FLAGS}}, sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
