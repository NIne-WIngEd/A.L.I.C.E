"""Guarded P2 entry for discovery or reviewed source-hash-map comparison.

Run ONLY through the existing same-owner fresh USER+NET namespace helper and
existing rayan-n0-base P2 container. This script creates no namespace, changes
no container mode, downloads nothing, and provides no fallback or privacy bypass.
Private path values remain unexamined until the actual process guard succeeds.
"""
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


# Import PUBLIC stdlib guard code only; no private path or source is read here.
_HELPER_PATH = Path(__file__).with_name("enter_private_network_namespace.py")
_SPEC = importlib.util.spec_from_file_location("personality_manifest_existing_namespace", _HELPER_PATH)
namespace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(namespace)


_FAILURE_STAGES = frozenset({
    "entry_invocation", "namespace_guard", "mode_validation", "compute_boundary",
    "public_code_and_run_admission", "code_commit_custody", "entry_location_custody",
    "public_dependencies", "public_pin_custody", "archive_input_admission",
    "member_map_admission", "reviewed_structure_admission", "member_map_custody",
    "member_map_schema", "manifest_discovery", "declared_hash_comparison",
    "result_schema", "member_map_recheck", "code_commit_recheck", "implementation_recheck",
    "result_binding", "receipt_seal", "member_map_close_recheck", "protected_structure_write",
    "public_summary_write"})
_AUDIT_REASON_CATEGORIES = {
    "audit_refused": "manifest_audit_refused",
    "audit_input_invalid": "source_validation_refused",
    "audit_custody_refused": "input_custody_refused",
    "audit_inventory_refused": "archive_inventory_refused",
    "audit_metadata_refused": "metadata_or_review_refused",
    "audit_structure_bound": "structure_resource_bound",
    "audit_code_changed": "implementation_changed"}
_FAILURE_CATEGORIES = frozenset({"unclassified_refusal", "boundary_or_schema_refused",
    "permission_refused", "required_input_absent", "create_only_conflict", "path_type_refused",
    "required_field_absent", "public_dependency_unavailable", "io_refused", "interrupted",
    *_AUDIT_REASON_CATEGORIES.values()})


class SafeAuditFailure(namespace.PrivateStageError):
    """Only fixed enums are public; original error args are never copied/logged."""

    def __init__(self, stage: str, category: str):
        super().__init__("manifest_structure_audit_refused")
        self.stage = stage if type(stage) is str and stage in _FAILURE_STAGES else "entry_invocation"
        self.category = category if type(category) is str and category in _FAILURE_CATEGORIES else "unclassified_refusal"


class _Progress:
    def __init__(self):
        self.stage = "entry_invocation"

    def mark(self, stage: str) -> None:
        if type(stage) is not str or stage not in _FAILURE_STAGES:
            raise SafeAuditFailure("entry_invocation", "unclassified_refusal")
        self.stage = stage


def _failure_category(exc: BaseException) -> str:
    # This function never stringifies errors or reads a source key/path value.
    # Only the already-imported, admitted module's exact fixed error labels may
    # distinguish custody/metadata/implementation refusal. Generic ValueError
    # text resembling one of those labels is not accepted as such a category.
    audit = sys.modules.get("src.alice_personality.n1.manifest_structure_audit")
    kind = getattr(audit, "ManifestStructureAuditError", None)
    if isinstance(kind, type) and isinstance(exc, kind):
        args = exc.args
        if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
            return _AUDIT_REASON_CATEGORIES.get(args[0], "unclassified_refusal")
        return "unclassified_refusal"
    raw = sys.modules.get("src.alice_personality.n1.raw_inference_lineage")
    kind = getattr(raw, "RawInferenceLineageError", None)
    if isinstance(kind, type) and isinstance(exc, kind):
        return "source_validation_refused"
    if isinstance(exc, namespace.PrivateStageError):
        return "boundary_or_schema_refused"
    for kind, category in ((PermissionError, "permission_refused"), (FileNotFoundError, "required_input_absent"),
                           (FileExistsError, "create_only_conflict"), (IsADirectoryError, "path_type_refused"),
                           (NotADirectoryError, "path_type_refused"), (ImportError, "public_dependency_unavailable"),
                           (KeyError, "required_field_absent"), (OSError, "io_refused"),
                           (KeyboardInterrupt, "interrupted"), (SystemExit, "interrupted")):
        if isinstance(exc, kind):
            return category
    return "unclassified_refusal"


def _failure_fields(exc: BaseException) -> dict:
    # Exact class/type and allowlists also protect callers that forge an error.
    stage, category = "entry_invocation", "unclassified_refusal"
    if type(exc) is SafeAuditFailure:
        fields = vars(exc)
        observed_stage, observed_category = fields.get("stage"), fields.get("category")
        if type(observed_stage) is str and observed_stage in _FAILURE_STAGES:
            stage = observed_stage
        if type(observed_category) is str and observed_category in _FAILURE_CATEGORIES:
            category = observed_category
    return {"failure_stage": stage, "failure_category": category}


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) != 0o600:
            raise namespace.PrivateStageError("protected artifact permissions differ")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_final_summary(summary: dict, initial: dict, protected_payload: bytes | None) -> None:
    extra = {"raw_member_map_file_sha256", "code_commit", "entry_implementation_sha256", "isolation", "receipt_sha256"}
    if protected_payload is not None:
        extra.add("protected_structure_file_sha256")
    valid = (type(summary) is dict and set(summary) == set(initial) | extra
             and namespace._canonical({key: summary[key] for key in initial}) == namespace._canonical(initial)
             and type(summary["code_commit"]) is str
             and re.fullmatch(r"[0-9a-f]{40}", summary["code_commit"]) is not None
             and type(summary["entry_implementation_sha256"]) is dict
             and set(summary["entry_implementation_sha256"]) == {"entry.py", "namespace.py"})
    if valid:
        digests = [summary["raw_member_map_file_sha256"], summary["receipt_sha256"],
                   *summary["entry_implementation_sha256"].values()]
        if protected_payload is not None:
            digests.append(summary["protected_structure_file_sha256"])
        valid = all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in digests)
    if valid:
        identity = summary["isolation"]
        valid = (type(identity) is dict and set(identity) == {"host_uid", "host_gid", "host_netns", "isolated_netns",
                    "interfaces", "p2_effective_uid", "p2_effective_gid"}
                 and type(identity["host_uid"]) is int and identity["host_uid"] == namespace.HOST_UID
                 and type(identity["host_gid"]) is int and identity["host_gid"] == namespace.HOST_GID
                 and type(identity["p2_effective_uid"]) is int and identity["p2_effective_uid"] in (0, namespace.HOST_UID)
                 and type(identity["p2_effective_gid"]) is int and identity["p2_effective_gid"] in (0, namespace.HOST_GID)
                 and identity["interfaces"] == ["lo"]
                 and all(type(identity[key]) is str and re.fullmatch(r"net:\[[0-9]+\]", identity[key])
                         for key in ("host_netns", "isolated_netns"))
                 and identity["host_netns"] != identity["isolated_netns"])
    if not valid or (protected_payload is not None and summary["protected_structure_file_sha256"] != \
                    sha256(protected_payload + b"\n").hexdigest()) or \
            summary["receipt_sha256"] != sha256(namespace._canonical(
                {key: value for key, value in summary.items() if key != "receipt_sha256"})).hexdigest():
        raise namespace.PrivateStageError("fixed audit summary binding differs")


def _run_structure_audit(*, compare_source_hashes: bool, progress: _Progress) -> dict:
    # REQUIRED FIRST OPERATION: actual namespace/identity, before private paths.
    # Recording a fixed literal stage performs no path/environment/source work.
    progress.mark("namespace_guard")
    isolation = namespace._p2_guard()
    progress.mark("mode_validation")
    if type(compare_source_hashes) is not bool:
        raise namespace.PrivateStageError("audit mode differs")
    os.umask(0o077)
    progress.mark("compute_boundary")
    root_value = Path(os.environ["COMPUTE_ROOT"])
    if not root_value.is_absolute() or ".." in root_value.parts or any(
            item.is_symlink() or getattr(item.lstat(), "st_file_attributes", 0) & 0x400
            for item in (root_value, *root_value.parents)):
        raise namespace.PrivateStageError("owner compute boundary is linked or ambiguous")
    root = root_value.resolve(strict=True)
    if not root.is_dir() or root.is_symlink() or root.stat().st_uid not in (0, namespace.HOST_UID) \
            or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise namespace.PrivateStageError("owner compute boundary differs")
    progress.mark("public_code_and_run_admission")
    code = namespace._bounded_path(os.environ["PERSONALITY_REPO_ROOT"], root, file=False)
    run = namespace._bounded_path(os.environ["PERSONALITY_RUN_ROOT"], root, file=False)
    if code == run or code in run.parents or run in code.parents or \
            run.stat().st_uid not in (0, namespace.HOST_UID) or \
            stat.S_IMODE(run.stat().st_mode) != 0o700 or any(run.iterdir()):
        raise namespace.PrivateStageError("audit output needs a fresh separate owner-only directory")
    revision = os.environ["PERSONALITY_EXPECTED_REVISION"]
    progress.mark("code_commit_custody")
    namespace._clean_code(code, revision)
    progress.mark("entry_location_custody")
    if namespace._bounded_path(str(Path(__file__).absolute()), root, file=True) != \
            code / "scripts/eipm/n1/audit_provenance_manifest_structure.py" or \
            namespace._bounded_path(str(_HELPER_PATH.absolute()), root, file=True) != \
            code / "scripts/eipm/n1/enter_private_network_namespace.py":
        raise namespace.PrivateStageError("running entry differs from admitted code")
    # Public imports are admitted only after the actual isolation and code pins.
    progress.mark("public_dependencies")
    sys.path.insert(0, str(code))
    from src.alice_personality.n1 import manifest_structure_audit as audit
    from src.alice_personality.n1.compiler import curated_frontier_v2_pin
    from src.alice_personality.n1 import raw_inference_lineage as raw
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise namespace.PrivateStageError("manifest audit must not import model runtimes")
    entry_code = {"entry.py": sha256(Path(__file__).read_bytes()).hexdigest(),
                  "namespace.py": sha256(_HELPER_PATH.read_bytes()).hexdigest()}
    progress.mark("public_pin_custody")
    pin = curated_frontier_v2_pin()
    if pin.archive_sha256 != namespace.PACKAGE_SHA256:
        raise namespace.PrivateStageError("public curated frontier archive pin differs")
    if compare_source_hashes and pin.members_sha256.get(
            pin.package_root + "/" + audit.LEGACY_RESERVE_PUBLIC_MEMBER) != audit.LEGACY_RESERVE_SHA256:
        raise namespace.PrivateStageError("public retained reserve member pin differs")
    progress.mark("archive_input_admission")
    package = namespace._bounded_path(os.environ["PRIVATE_PACKAGE_PATH"], root, file=True)
    raw_source = namespace._bounded_path(os.environ["PERSONALITY_RAW_SOURCE_PATH"], root, file=True)
    progress.mark("member_map_admission")
    member_map_path = namespace._bounded_path(os.environ["PERSONALITY_RAW_MEMBER_MAP_PATH"], root, file=True)
    sources = (package, raw_source, member_map_path)
    if len(set(sources)) != 3 or any(code in path.parents or run in path.parents for path in sources):
        raise namespace.PrivateStageError("audit inputs must be separate from code and new outputs")
    if stat.S_IMODE(member_map_path.stat().st_mode) != 0o600 \
            or member_map_path.stat().st_uid not in (0, namespace.HOST_UID):
        raise namespace.PrivateStageError("protected map permissions or owner differs")
    map_sha = os.environ["PERSONALITY_RAW_MEMBER_MAP_SHA256"]
    reviewed_structure = None
    if compare_source_hashes:
        progress.mark("reviewed_structure_admission")
        reviewed_structure = namespace._bounded_path(os.environ["PERSONALITY_REVIEWED_STRUCTURE_PATH"], root, file=True)
        if reviewed_structure in sources or code in reviewed_structure.parents or run in reviewed_structure.parents or \
                stat.S_IMODE(reviewed_structure.stat().st_mode) != 0o600 or \
                reviewed_structure.stat().st_uid not in (0, namespace.HOST_UID):
            raise namespace.PrivateStageError("reviewed protected structure location or permissions differs")
    summary_path = run / ("declared-source-hash-comparison.json" if compare_source_hashes else "manifest-structure-summary.json")
    structure_path = run / "manifest-structure.protected.json"
    progress.mark("member_map_custody")
    with audit._PinnedFile(member_map_path, map_sha, namespace.MAX_MEMBER_MAP_BYTES) as pinned_map:
        payload = pinned_map.bounded_bytes()
        progress.mark("member_map_schema")
        member_map = namespace._raw_member_map(raw._json(payload))
        # Capturing output prevents any arbitrary metadata/error text reaching logs.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            if compare_source_hashes:
                progress.mark("declared_hash_comparison")
                summary = audit.compare_declared_source_hashes(package, curated_pin=pin,
                    raw_archive_path=raw_source, expected_raw_archive_sha256=namespace.RAW_SOURCE_SHA256,
                    raw_member_map=member_map, reviewed_structure_path=reviewed_structure,
                    expected_reviewed_structure_file_sha256=audit.REVIEWED_STRUCTURE_FILE_SHA256,
                    expected_curated_manifest_sha256=audit.REVIEWED_CURATED_MANIFEST_SHA256,
                    expected_raw_generation_manifest_sha256=audit.REVIEWED_RAW_MANIFEST_SHA256)
                progress.mark("result_schema")
                audit.verify_comparison_result(summary)
            else:
                progress.mark("manifest_discovery")
                protected, summary = audit.discover_manifest_structure(package, curated_pin=pin,
                    raw_archive_path=raw_source, expected_raw_archive_sha256=namespace.RAW_SOURCE_SHA256,
                    raw_member_map=member_map)
                progress.mark("result_schema")
                audit.verify_discovery_result(protected, summary)
        progress.mark("member_map_recheck")
        pinned_map.verify()
        progress.mark("code_commit_recheck")
        namespace._clean_code(code, revision)
        progress.mark("implementation_recheck")
        if entry_code != {"entry.py": sha256(Path(__file__).read_bytes()).hexdigest(),
                          "namespace.py": sha256(_HELPER_PATH.read_bytes()).hexdigest()} \
                or audit.implementation_hashes() != summary["implementation_sha256"]:
            raise namespace.PrivateStageError("audit implementation changed")
        # No files are created until every source/map/code recheck succeeds.
        progress.mark("result_binding")
        structure_payload = None if compare_source_hashes else raw._canonical(protected)
        if structure_payload is not None and sha256(structure_payload).hexdigest() != summary["protected_structure_sha256"]:
            raise namespace.PrivateStageError("protected structure binding differs")
        initial_summary = dict(summary)
        summary.update(raw_member_map_file_sha256=map_sha, code_commit=revision,
                       entry_implementation_sha256=entry_code, isolation=isolation)
        if structure_payload is not None:
            summary["protected_structure_file_sha256"] = sha256(structure_payload + b"\n").hexdigest()
        progress.mark("receipt_seal")
        summary["receipt_sha256"] = sha256(raw._canonical(summary)).hexdigest()
        _validate_final_summary(summary, initial_summary, structure_payload)
        progress.mark("member_map_close_recheck")
    if structure_payload is not None:
        progress.mark("protected_structure_write")
        _write_new(structure_path, structure_payload + b"\n")
    progress.mark("public_summary_write")
    _write_new(summary_path, raw._canonical(summary) + b"\n")
    return summary


def run_structure_audit(*, compare_source_hashes: bool = False) -> dict:
    """Run unchanged custody checks; failures retain only fixed stage/category."""
    progress = _Progress()
    try:
        return _run_structure_audit(compare_source_hashes=compare_source_hashes, progress=progress)
    except BaseException as exc:
        raise SafeAuditFailure(progress.stage, _failure_category(exc)) from None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    comparison = args == ["--p2-source-hash-comparison-stage"]
    if args != ["--p2-structure-stage"] and not comparison:
        print("usage: guarded manifest audit --p2-structure-stage | --p2-source-hash-comparison-stage", file=sys.stderr)
        return 2
    try:
        summary = run_structure_audit(compare_source_hashes=True) if comparison else run_structure_audit()
        public = ("schema", "state", "receipt_sha256", "curated_manifest_sha256", "raw_generation_manifest_sha256", "link_compared",
                  "acceptance_authority", "private_gradient_authorized", "historical_authority_granted",
                  "training_authorized")
        public += (("declared_source_hash_count", "reserve_digest_match_count", "generator_declares_matching_digest",
                    "original_source_independently_verified", "legacy_namespace_proven", "source_admission")
                   if comparison else ("protected_structure_file_sha256",))
        print(json.dumps({key: summary[key] for key in public}, sort_keys=True))
        return 0
    except BaseException as exc:
        # No arbitrary exception type/text, keys, paths or traceback is public.
        print(json.dumps({"state": "FAILED_UNQUALIFIED", "failure_reason": "manifest_structure_audit_refused",
                          **_failure_fields(exc),
                          "link_compared": False, "acceptance_authority": False,
                          "private_gradient_authorized": False, "historical_authority_granted": False,
                          "training_authorized": False, "source_admission": False,
                          "original_source_independently_verified": False, "legacy_namespace_proven": False},
                         sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
