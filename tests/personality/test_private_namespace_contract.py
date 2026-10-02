"""Public stand-ins only: namespace refusals, custody order and safe receipts.

These tests never enter a real namespace, contact Magnolia or open private
sources. Mock compilation checks launcher mechanics, not source qualification.
"""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts/eipm/n1/enter_private_network_namespace.py"
SBATCH = REPO / "scripts/eipm/n1/magnolia_compile_identity_substrate.sbatch"
SPEC = importlib.util.spec_from_file_location("personality_private_namespace_public_test", HELPER)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.mark.parametrize("message, category", [
    ("active source kind and provenance class disagree", "source_provenance_kind_mismatch"),
    ("internal checksums do not cover the complete archive membership", "source_checksum_inventory_mismatch"),
    ("canonical E0 requires an explicit historical truth flag", "source_history_authority_missing"),
    ("curation manifest count or closed-gap status mismatch", "source_manifest_count_mismatch"),
    ("training_authority must be a Boolean", "source_row_boolean_type_mismatch"),
    ("recommended_supervision_lane must be an array of nonempty strings", "source_row_list_type_mismatch"),
    ("personality dimensions contains duplicate identifiers or labels", "source_row_list_duplicate"),
    ("E0 text requires a SHA256 digest", "source_text_pin_missing_or_invalid"),
    ("context-only or excluded evidence cannot enable identity loss", "source_loss_role_conflict"),
    ("explicitly excluded evidence cannot enable identity loss", "source_loss_role_conflict"),
    ("FICTITIOUS SOURCE SENTENCE MUST NOT REACH LOGS", "private_stage_refused"),
])
def test_failure_categories_are_fixed_public_diagnostics(message, category):
    assert helper._safe_reason(ValueError(message)) == category


def test_raw_source_failure_categories_admit_only_fixed_reader_labels():
    from src.alice_personality.n1.raw_inference_lineage import RawInferenceLineageError
    assert helper._safe_reason(RawInferenceLineageError("logical_member_root_mismatch")) == "raw_lineage_logical_member_root_mismatch"
    assert helper._safe_reason(RawInferenceLineageError("PRIVATE SOURCE SENTENCE")) == "private_stage_refused"
    assert helper._safe_reason(ValueError("logical_member_root_mismatch")) == "private_stage_refused"


@pytest.fixture
def namespace(monkeypatch):
    monkeypatch.setenv("PERSONALITY_HOST_NETNS", "net:[100]")
    monkeypatch.setenv("PERSONALITY_ISOLATED_NETNS", "net:[200]")
    monkeypatch.setenv("PERSONALITY_HOST_UID", "1905")
    monkeypatch.setenv("PERSONALITY_HOST_GID", "100")
    monkeypatch.setattr(helper.sys, "platform", "linux")
    for name, value in {"getuid": 1905, "geteuid": 1905, "getgid": 100, "getegid": 100}.items():
        monkeypatch.setattr(helper.os, name, lambda value=value: value, raising=False)
    monkeypatch.setattr(helper.os, "readlink", lambda path: "net:[200]")
    monkeypatch.setattr(helper.socket, "if_nameindex", lambda: [(1, "lo")])
    monkeypatch.setattr(helper, "_read_map", lambda path: ["1905", "1905", "1"]
                        if path.endswith("uid_map") else ["100", "100", "1"])
    monkeypatch.setattr(helper, "_close_inherited_descriptors", Mock())


def test_same_uid_mapping_and_combined_namespace_before_exec(namespace, monkeypatch):
    unshare = Mock(return_value=0)
    library = SimpleNamespace(unshare=unshare)
    monkeypatch.setattr(helper.ctypes, "CDLL", Mock(return_value=library))
    writes = Mock()
    monkeypatch.setattr(helper, "_write_once", writes)
    namespaces = iter(["net:[100]", "net:[200]"])
    monkeypatch.setattr(helper.os, "readlink", lambda path: next(namespaces))
    monkeypatch.setenv("https_proxy", "http://public-fixture.invalid")
    execv = Mock(side_effect=RuntimeError("public test stops at exec"))
    monkeypatch.setattr(helper.os, "execv", execv)
    assert helper.main(["--", "/bin/bash", "-c", "true"]) == 3
    unshare.assert_called_once_with(helper.CLONE_NEWUSER | helper.CLONE_NEWNET)
    assert writes.call_args_list == [
        (("/proc/self/uid_map", "1905 1905 1\n"),),
        (("/proc/self/setgroups", "deny\n"),),
        (("/proc/self/gid_map", "100 100 1\n"),),
    ]
    helper._close_inherited_descriptors.assert_called_once()
    execv.assert_called_once_with("/bin/bash", ["/bin/bash", "-c", "true"])
    assert "https_proxy" not in helper.os.environ


@pytest.mark.parametrize("failure", ["root", "unshare", "host_netns", "extra_interface", "wrong_map"])
def test_namespace_failure_has_no_exec_or_source_resolution(namespace, monkeypatch, failure, capsys):
    unshare = Mock(return_value=-1 if failure == "unshare" else 0)
    monkeypatch.setattr(helper.ctypes, "CDLL", Mock(return_value=SimpleNamespace(unshare=unshare)))
    monkeypatch.setattr(helper, "_write_once", Mock())
    monkeypatch.setattr(helper.os, "readlink", Mock(side_effect=["net:[100]", "net:[200]"]))
    if failure == "root":
        monkeypatch.setattr(helper.os, "geteuid", lambda: 0)
    elif failure == "host_netns":
        monkeypatch.setattr(helper.os, "readlink", lambda path: "net:[999]")
    elif failure == "extra_interface":
        monkeypatch.setattr(helper.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    elif failure == "wrong_map":
        monkeypatch.setattr(helper, "_read_map", lambda path: ["0", "1905", "1"])
    execv, bounded = Mock(), Mock()
    monkeypatch.setattr(helper.os, "execv", execv)
    monkeypatch.setattr(helper, "_bounded_path", bounded)
    assert helper.main(["--", "/bin/bash", "-c", "true"]) == 3
    execv.assert_not_called()
    bounded.assert_not_called()
    if failure == "root":
        unshare.assert_not_called()
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["state"] == "FAILED_UNQUALIFIED"
    assert set(diagnostic) == {"state", "failure_type", "failure_reason"}


@pytest.mark.parametrize("failure", ["host_netns", "extra_interface", "wrong_map", "wrong_uid"])
def test_actual_p2_refusal_precedes_every_source_path_operation(namespace, monkeypatch, failure):
    if failure == "host_netns":
        monkeypatch.setattr(helper.os, "readlink", lambda path: "net:[100]")
    elif failure == "extra_interface":
        monkeypatch.setattr(helper.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    elif failure == "wrong_map":
        monkeypatch.setattr(helper, "_read_map", lambda path: ["0", "1905", "1"])
    else:
        monkeypatch.setattr(helper.os, "geteuid", lambda: 1000)
    bounded, hashed, imported = Mock(), Mock(), Mock()
    monkeypatch.setattr(helper, "_bounded_path", bounded)
    monkeypatch.setattr(helper, "_file_hash", hashed)
    monkeypatch.setattr(helper, "_clean_code", imported)
    with pytest.raises(helper.PrivateStageError):
        helper.run_private_compile()
    bounded.assert_not_called()
    hashed.assert_not_called()
    imported.assert_not_called()


def test_actual_p2_allows_emulated_root_with_original_host_maps(namespace, monkeypatch):
    monkeypatch.setattr(helper.os, "geteuid", lambda: 0)
    monkeypatch.setattr(helper.os, "getegid", lambda: 0)
    observed = helper._p2_guard()
    assert observed["host_uid"] == 1905 and observed["host_gid"] == 100
    assert observed["p2_effective_uid"] == 0 and observed["p2_effective_gid"] == 0
    assert observed["interfaces"] == ["lo"]
    assert observed["isolated_netns"] != observed["host_netns"]


def test_inherited_descriptors_and_outbound_environment_are_removed(monkeypatch):
    monkeypatch.setattr(helper.os, "listdir", lambda path: ["0", "1", "2", "7", "8", "not-a-fd"])
    close = Mock()
    monkeypatch.setattr(helper.os, "close", close)
    helper._close_inherited_descriptors()
    assert [call.args[0] for call in close.call_args_list] == [7, 8]
    for key in ("HTTPS_PROXY", "all_proxy", "No_Proxy", "SSH_AUTH_SOCK", "BASH_ENV", "ENV", "BASH_FUNC_x%%"):
        monkeypatch.setenv(key, "public-fixture")
    monkeypatch.setenv("PERSONALITY_REPO_ROOT", "/public-code")
    helper._strip_environment()
    assert helper.os.environ["PERSONALITY_REPO_ROOT"] == "/public-code"
    assert not any("proxy" in key.lower() or key.startswith("BASH_FUNC_") for key in helper.os.environ)
    assert "SSH_AUTH_SOCK" not in helper.os.environ and "BASH_ENV" not in helper.os.environ


def test_descriptor_closure_fails_closed_except_closed_fd_race(monkeypatch):
    monkeypatch.setattr(helper.os, "listdir", lambda path: ["8"])
    monkeypatch.setattr(helper.os, "close", Mock(side_effect=OSError(helper.errno.EBADF, "fixture")))
    helper._close_inherited_descriptors()
    monkeypatch.setattr(helper.os, "close", Mock(side_effect=OSError(helper.errno.EPERM, "fixture")))
    with pytest.raises(helper.PrivateStageError, match="descriptor closure failed"):
        helper._close_inherited_descriptors()


@pytest.fixture
def public_stage(tmp_path, monkeypatch):
    """A fabricated public archive pathname and fabricated compiler receipt."""
    code, run, archive = tmp_path / "public-code", tmp_path / "run", tmp_path / "public-fixture.zip"
    code.mkdir()
    run.mkdir(mode=0o700)
    # The fake P2 guard above is a stand-in, so bind the public temporary
    # filesystem owner here. Production retains Magnolia's exact UID 1905.
    # GitHub's runner UID differs from both Windows and Magnolia.
    monkeypatch.setattr(helper, "HOST_UID", run.stat().st_uid)
    archive.write_bytes(b"public test stand-in, never a real curated archive")
    for key, value in {"COMPUTE_ROOT": tmp_path, "PERSONALITY_REPO_ROOT": code,
                       "PERSONALITY_RUN_ROOT": run, "PRIVATE_PACKAGE_PATH": archive,
                       "PERSONALITY_EXPECTED_REVISION": "a" * 40}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv("PERSONALITY_RAW_LINEAGE_PATH", raising=False)
    monkeypatch.delenv("PERSONALITY_RAW_LINEAGE_SHA256", raising=False)
    monkeypatch.delenv("PERSONALITY_RAW_SOURCE_PATH", raising=False)
    monkeypatch.delenv("PERSONALITY_RAW_MEMBER_MAP_PATH", raising=False)
    monkeypatch.delenv("PERSONALITY_RAW_MEMBER_MAP_SHA256", raising=False)
    guard = Mock(return_value={"host_uid": 1905, "host_gid": 100, "host_netns": "net:[100]",
                               "isolated_netns": "net:[200]", "interfaces": ["lo"]})
    monkeypatch.setattr(helper, "_p2_guard", guard)
    clean = Mock()
    monkeypatch.setattr(helper, "_clean_code", clean)
    # Windows does not implement POSIX ownership/permissions; fixture metadata
    # supplies the declared POSIX modes. chmod calls are still exercised.
    if sys.platform == "win32":
        monkeypatch.setattr(helper.stat, "S_IMODE", lambda mode: 0o700 if stat.S_ISDIR(mode) else 0o600)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    receipt = {"state": "COMPILED_UNQUALIFIED", "receipt_sha256": "b" * 64,
               "private_gradient_authorized": False, "acceptance_authority": False,
               "active_record_count": 3, "source_family_count": 2,
               "kind_counts": {"E0": 1, "ASYN_DIRECT": 2}, "split_counts": {"TRAIN": 2, "DEV": 1},
               "files": [{"path": "public-fixture.jsonl", "bytes": 3, "sha256": sha256(b"{}\n").hexdigest()}]}

    def compile_public_stand_in(package, output, **kwargs):
        output.mkdir(mode=0o700)
        (output / "public-fixture.jsonl").write_bytes(b"{}\n")
        print("FICTITIOUS SOURCE SENTENCE MUST NOT REACH LOGS")
        return dict(receipt)

    compile_call = Mock(side_effect=compile_public_stand_in)
    verify_call = Mock(side_effect=lambda output, **kwargs: dict(receipt))
    pin = Mock(return_value=SimpleNamespace(archive_sha256=helper.PACKAGE_SHA256))
    monkeypatch.setitem(sys.modules, "src.alice_personality.n1.compiler",
                        SimpleNamespace(compile_package=compile_call, verify_compiled=verify_call,
                                        curated_frontier_v2_pin=pin))
    return SimpleNamespace(code=code, run=run, archive=archive, guard=guard, clean=clean,
                           compile=compile_call, verify=verify_call, pin=pin, receipt=receipt)


def test_compilation_bound_and_append_only_unqualified_receipt(public_stage, capsys):
    stage = public_stage
    receipt = helper.run_private_compile()
    assert receipt["state"] == "COMPILED_UNQUALIFIED"
    for key in ("private_gradient_authorized", "acceptance_authority", "model_training_performed", "weights_created"):
        assert receipt[key] is False
    assert receipt["behavior_qualification"] is None
    assert receipt["source_package_sha256"] == helper.PACKAGE_SHA256
    assert receipt["source_commit"] == "a" * 40
    assert receipt["compiled_receipt_sha256"] == "b" * 64
    stage.guard.assert_called_once()
    assert stage.clean.call_count == 2
    stage.verify.assert_called_once_with(stage.run / "compiled-substrate",
                                        expected_source_archive_sha256=helper.PACKAGE_SHA256,
                                        expected_receipt_sha256="b" * 64)
    assert capsys.readouterr().out == ""
    path = stage.run / "compile_summary.json"
    frozen = path.read_bytes()
    parsed = json.loads(frozen)
    digest = parsed.pop("receipt_sha256")
    assert digest == sha256(helper._canonical(parsed)).hexdigest()
    with pytest.raises(helper.PrivateStageError, match="fresh"):
        helper.run_private_compile()
    assert path.read_bytes() == frozen
    assert stage.compile.call_count == 1


@pytest.mark.parametrize("formatted", [False, True])
def test_pinned_raw_lineage_is_optional_bounded_and_checked_before_compile(public_stage, monkeypatch, formatted):
    path = public_stage.run.parent / "public-registry.json"
    payload = b'{"EINF":["PUBLIC_RAW_1"]}'
    if formatted:
        payload = b'{\n  "EINF": ["PUBLIC_RAW_1"]\n}\n'
    path.write_bytes(payload)
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_PATH", str(path))
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_SHA256", sha256(payload).hexdigest())
    receipt = helper.run_private_compile()
    assert public_stage.compile.call_args.kwargs["raw_lineage_registry"] == {"EINF": ["PUBLIC_RAW_1"]}
    assert receipt["raw_lineage_file_sha256"] == sha256(payload).hexdigest()


@pytest.mark.parametrize(("failure", "selected"),
    [(failure, selected) for failure in (None, "extra_metadata", "bad_metadata_seal", "original_source_mutation")
     for selected in (False, True)] + [("member_map_mutation", True)])
def test_original_raw_source_derives_protected_registry_in_same_isolated_stage(public_stage, monkeypatch, capsys, failure, selected):
    raw = public_stage.run.parent / "original-public-source.zip"
    raw.write_bytes(b"original independently pinned public raw source stand-in")
    source_pin = sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setattr(helper, "RAW_SOURCE_SHA256", source_pin)
    monkeypatch.setenv("PERSONALITY_RAW_SOURCE_PATH", str(raw))
    registry = {"EINF": ["PUBLIC_RAW_1"]}
    registry_pin = sha256(helper._canonical(registry)).hexdigest()
    raw_receipt = {"schema": "alice-personality-raw-inference-lineage-v1", "state": "DERIVED_UNQUALIFIED",
        "source_archive_sha256": source_pin, "source_member_path": "public/einf_proposals.jsonl",
        "source_member_sha256": "c" * 64, "source_member_bytes": 40,
        "generation_manifest_path": "public/generation_manifest.json", "generation_manifest_sha256": "d" * 64,
        "checksum_ledger_path": "public/SHA256SUMS.txt", "checksum_ledger_sha256": "e" * 64,
        "code_sha256": "f" * 64, "mechanical_fixture_only": False,
        "training_authorized": False, "behavior_qualification": None, "registry_sha256": registry_pin,
        "raw_inference_count": 1, "acceptance_authority": False,
        "private_gradient_authorized": False, "historical_authority_granted": False}
    raw_receipt["receipt_sha256"] = sha256(helper._canonical(raw_receipt)).hexdigest()
    if selected:
        raw_receipt["source_member_path"] = "public/explicit_inference_fixture.jsonl"
        raw_receipt.pop("receipt_sha256")
        raw_receipt["receipt_sha256"] = sha256(helper._canonical(raw_receipt)).hexdigest()
        member_map = _member_map_fixture(source_pin)
        member_path = public_stage.run.parent / "protected-public-member-map.json"
        member_payload = helper._canonical(member_map)
        member_path.write_bytes(member_payload)
        monkeypatch.setenv("PERSONALITY_RAW_MEMBER_MAP_PATH", str(member_path))
        monkeypatch.setenv("PERSONALITY_RAW_MEMBER_MAP_SHA256", sha256(member_payload).hexdigest())
    if failure == "extra_metadata":
        raw_receipt.pop("receipt_sha256")
        raw_receipt["PRIVATE_LOOKING_EXTRA_METADATA"] = "FICTITIOUS SOURCE TEXT MUST NOT LEAVE SUMMARY"
        raw_receipt["receipt_sha256"] = sha256(helper._canonical(raw_receipt)).hexdigest()
    if failure == "bad_metadata_seal":
        raw_receipt["receipt_sha256"] = "0" * 64
    def derive(path, *, expected_archive_sha256, source_member_path=None):
        assert public_stage.guard.call_count == 1
        assert path == raw and expected_archive_sha256 == source_pin
        assert source_member_path == (member_map["source_member_path"] if selected else None)
        print("FICTITIOUS RAW SOURCE TEXT MUST NOT BE LOGGED")
        return registry, raw_receipt
    call = Mock(side_effect=derive)
    monkeypatch.setitem(sys.modules, "src.alice_personality.n1.raw_inference_lineage",
                        SimpleNamespace(derive_raw_inference_registry=call))
    if failure in {"original_source_mutation", "member_map_mutation"}:
        original = public_stage.compile.side_effect
        def mutate(package, output, **kwargs):
            receipt = original(package, output, **kwargs)
            if failure == "original_source_mutation":
                raw.write_bytes(b"X" * raw.stat().st_size)
            else:
                member_path.write_bytes(member_payload + b" ")
            return receipt
        public_stage.compile.side_effect = mutate
    if failure is not None:
        with pytest.raises(helper.PrivateStageError, match="sanitized"):
            helper.run_private_compile()
        assert capsys.readouterr().out == ""
        receipt = json.loads((public_stage.run / "compile_summary.json").read_bytes())
        assert receipt["state"] == "FAILED_UNQUALIFIED"
        assert "PUBLIC_RAW_1" not in (public_stage.run / "compile_summary.json").read_text()
        assert "FICTITIOUS SOURCE TEXT" not in (public_stage.run / "compile_summary.json").read_text()
        if failure not in {"original_source_mutation", "member_map_mutation"}:
            public_stage.compile.assert_not_called()
            assert "raw_inference_source" not in receipt
        return
    receipt = helper.run_private_compile()
    assert capsys.readouterr().out == ""
    call.assert_called_once()
    assert public_stage.compile.call_args.kwargs["raw_lineage_registry"] == registry
    assert receipt["raw_lineage_file_sha256"] == registry_pin
    assert receipt["raw_inference_source"]["source_receipt_sha256"] == raw_receipt["receipt_sha256"]
    assert receipt["raw_inference_source"]["raw_inference_count"] == 1
    assert "source_member_path" not in receipt["raw_inference_source"]
    assert (public_stage.run / "raw-inference-registry.json").read_bytes() == helper._canonical(registry)
    assert json.loads((public_stage.run / "raw-inference-source.json").read_bytes()) == raw_receipt
    assert "PUBLIC_RAW_1" not in (public_stage.run / "compile_summary.json").read_text()
    assert "explicit_inference_fixture" not in (public_stage.run / "compile_summary.json").read_text()
    if selected:
        assert (public_stage.run / "raw-member-map.json").read_bytes() == member_payload
        assert receipt["raw_member_map_file_sha256"] == sha256(member_payload).hexdigest()
    assert receipt["acceptance_authority"] is False and receipt["private_gradient_authorized"] is False


def _member_map_fixture(source_pin):
    return {"schema": "alice-personality-raw-inference-source-layout-review-v1", "state": "PROPOSED_UNQUALIFIED",
        "source_archive_sha256": source_pin, "source_member_path": "public/explicit_inference_fixture.jsonl",
        "generation_manifest_path": "public/generation_manifest.json", "checksum_ledger_path": "public/SHA256SUMS.txt",
        "source_layout_receipt_file_sha256": "1" * 64, "source_layout_receipt_sha256": "2" * 64,
        "selection_basis": "PUBLIC fixture directory metadata only; proposal schema unverified",
        "acceptance_authority": False, "private_gradient_authorized": False,
        "historical_authority_granted": False, "training_authorized": False}


@pytest.mark.parametrize("failure", ["missing_pin", "without_source", "external_pin", "source_pin",
    "authority", "duplicate_key", "unsafe_member", "different_root", "metadata_alias", "oversize", "in_code"])
def test_explicit_member_map_refuses_unbound_or_authoritative_input_before_source_read(public_stage, monkeypatch, failure):
    raw = public_stage.run.parent / "unopened-public-raw-source.zip"
    raw.write_bytes(b"PUBLIC source sentinel must remain unopened by reader")
    source_pin = sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setattr(helper, "RAW_SOURCE_SHA256", source_pin)
    monkeypatch.setenv("PERSONALITY_RAW_SOURCE_PATH", str(raw))
    value = _member_map_fixture(source_pin)
    if failure == "source_pin":
        value["source_archive_sha256"] = "0" * 64
    if failure == "authority":
        value["training_authorized"] = True
    if failure == "unsafe_member":
        value["source_member_path"] = "../private-looking-name.jsonl"
    if failure == "different_root":
        value["source_member_path"] = "other/public-fixture.jsonl"
    if failure == "metadata_alias":
        value["source_member_path"] = value["checksum_ledger_path"]
    payload = helper._canonical(value)
    if failure == "duplicate_key":
        payload = payload[:-1] + b',"training_authorized":false}'
    if failure == "oversize":
        payload += b" " * helper.MAX_MEMBER_MAP_BYTES
    path = (public_stage.code if failure == "in_code" else public_stage.run.parent) / "public-member-map.json"
    path.write_bytes(payload)
    monkeypatch.setenv("PERSONALITY_RAW_MEMBER_MAP_PATH", str(path))
    monkeypatch.setenv("PERSONALITY_RAW_MEMBER_MAP_SHA256", "0" * 64 if failure == "external_pin" else sha256(payload).hexdigest())
    if failure == "missing_pin":
        monkeypatch.delenv("PERSONALITY_RAW_MEMBER_MAP_SHA256")
    if failure == "without_source":
        monkeypatch.delenv("PERSONALITY_RAW_SOURCE_PATH")
    derive = Mock(side_effect=AssertionError("reader must not open source"))
    monkeypatch.setitem(sys.modules, "src.alice_personality.n1.raw_inference_lineage", SimpleNamespace(derive_raw_inference_registry=derive))
    with pytest.raises(helper.PrivateStageError, match="sanitized"):
        helper.run_private_compile()
    derive.assert_not_called()
    public_stage.compile.assert_not_called()
    summary = (public_stage.run / "compile_summary.json").read_text()
    assert "private-looking-name" not in summary and "explicit_inference_fixture" not in summary


def test_raw_source_and_supplied_registry_are_ambiguous_and_refused(public_stage, monkeypatch):
    monkeypatch.setenv("PERSONALITY_RAW_SOURCE_PATH", str(public_stage.run.parent / "unopened-original.zip"))
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_PATH", str(public_stage.run.parent / "unopened-registry.json"))
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_SHA256", "c" * 64)
    with pytest.raises(helper.PrivateStageError, match="sanitized"):
        helper.run_private_compile()
    public_stage.compile.assert_not_called()
    assert json.loads((public_stage.run / "compile_summary.json").read_bytes())["failure_phase"] == "private_source_custody"


def test_actual_source_layout_names_stay_in_protected_failure_metadata(public_stage, monkeypatch, capsys):
    raw = public_stage.run.parent / "public-source.zip"
    raw.write_bytes(b"independently pinned fixture source")
    monkeypatch.setenv("PERSONALITY_RAW_SOURCE_PATH", str(raw))
    error_type = type("RawInferenceLineageError", (ValueError,), {})
    layout = {"schema": "alice-personality-raw-inference-source-layout-v1", "state": "UNQUALIFIED",
              "source_archive_sha256": helper.RAW_SOURCE_SHA256,
              "member_paths": ["PROTECTED_FIXTURE_ARTIFACT_NAME.jsonl"], "member_count": 1}
    derive = Mock(side_effect=error_type("missing_or_ambiguous_logical_member"))
    audit = Mock(return_value=layout)
    monkeypatch.setitem(sys.modules, "src.alice_personality.n1.raw_inference_lineage",
                        SimpleNamespace(derive_raw_inference_registry=derive, audit_raw_inference_source_layout=audit))
    with pytest.raises(helper.PrivateStageError, match="sanitized"):
        helper.run_private_compile()
    assert capsys.readouterr().out == ""
    public_stage.compile.assert_not_called()
    audit.assert_called_once_with(raw, expected_archive_sha256=helper.RAW_SOURCE_SHA256)
    summary_bytes = (public_stage.run / "compile_summary.json").read_bytes()
    receipt = json.loads(summary_bytes)
    protected = (public_stage.run / "raw-source-layout.json").read_bytes()
    assert json.loads(protected) == layout
    assert receipt["raw_source_layout_file_sha256"] == sha256(protected).hexdigest()
    assert b"PROTECTED_FIXTURE_ARTIFACT_NAME" not in summary_bytes
    assert receipt["failure_reason"] == "raw_lineage_missing_or_ambiguous_logical_member"


def test_supplied_registry_change_during_compile_cannot_publish_success(public_stage, monkeypatch):
    path = public_stage.run.parent / "public-registry.json"
    payload = b'{"EINF":["PUBLIC_RAW_1"]}'
    path.write_bytes(payload)
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_PATH", str(path))
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_SHA256", sha256(payload).hexdigest())
    original = public_stage.compile.side_effect
    def mutate(package, output, **kwargs):
        receipt = original(package, output, **kwargs)
        path.write_bytes(b'{"EINF":["PUBLIC_RAW_2"]}')
        return receipt
    public_stage.compile.side_effect = mutate
    with pytest.raises(helper.PrivateStageError, match="sanitized"):
        helper.run_private_compile()
    receipt = json.loads((public_stage.run / "compile_summary.json").read_bytes())
    assert receipt["state"] == "FAILED_UNQUALIFIED" and receipt["failure_phase"] == "publication_checks"


def test_in_memory_registry_change_during_compile_cannot_publish_success(public_stage, monkeypatch):
    path = public_stage.run.parent / "public-registry.json"
    payload = b'{"EINF":["PUBLIC_RAW_1"]}'
    path.write_bytes(payload)
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_PATH", str(path))
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_SHA256", sha256(helper._canonical({"EINF": ["PUBLIC_RAW_1"]})).hexdigest())
    # Use canonical bytes so the external file binding matches its content.
    path.write_bytes(helper._canonical({"EINF": ["PUBLIC_RAW_1"]}))
    original = public_stage.compile.side_effect
    def mutate(package, output, **kwargs):
        receipt = original(package, output, **kwargs)
        kwargs["raw_lineage_registry"]["EINF"].append("PUBLIC_RAW_2")
        return receipt
    public_stage.compile.side_effect = mutate
    with pytest.raises(helper.PrivateStageError, match="sanitized"):
        helper.run_private_compile()
    receipt = json.loads((public_stage.run / "compile_summary.json").read_bytes())
    assert receipt["state"] == "FAILED_UNQUALIFIED" and receipt["failure_phase"] == "publication_checks"
    assert path.read_bytes() == helper._canonical({"EINF": ["PUBLIC_RAW_1"]})


@pytest.mark.parametrize("failure", ["wrong_pin", "too_large", "missing_pair", "inside_code", "duplicate_keys", "nonfinite"])
def test_raw_lineage_refusal_has_no_compilation(public_stage, monkeypatch, failure):
    path = public_stage.code / "public-registry.json" if failure == "inside_code" else public_stage.run.parent / "public-registry.json"
    payload = b'{"EINF":["PUBLIC_RAW_1"]}'
    if failure == "duplicate_keys":
        payload = b'{"EINF":["PUBLIC_RAW_1"],"EINF":["PUBLIC_RAW_2"]}'
    elif failure == "nonfinite":
        payload = b'{"EINF":[NaN]}'
    path.write_bytes(payload)
    monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_PATH", str(path))
    if failure != "missing_pair":
        monkeypatch.setenv("PERSONALITY_RAW_LINEAGE_SHA256", "0" * 64 if failure == "wrong_pin" else sha256(payload).hexdigest())
    if failure == "too_large":
        monkeypatch.setattr(helper, "MAX_REGISTRY_BYTES", 1)
    with pytest.raises(helper.PrivateStageError, match="sanitized"):
        helper.run_private_compile()
    public_stage.compile.assert_not_called()
    assert json.loads((public_stage.run / "compile_summary.json").read_bytes())["state"] == "FAILED_UNQUALIFIED"


def test_raw_hash_bound_is_enforced_even_if_stat_claims_small_file(tmp_path):
    path = tmp_path / "public-growing-registry.json"
    path.write_bytes(b"{}" * 100)

    class GrowingPublicFixture:
        def stat(self):
            return SimpleNamespace(st_size=1)

        def open(self, mode):
            return path.open(mode)

    with pytest.raises(helper.PrivateStageError, match="byte bound"):
        helper._file_hash(GrowingPublicFixture(), max_bytes=16)


@pytest.mark.parametrize("failure", ["not_empty", "inside_code", "wrong_mode"])
def test_invalid_output_root_precedes_private_source_access(public_stage, monkeypatch, failure):
    if failure == "not_empty":
        (public_stage.run / "earlier-public-receipt.json").write_text("{}")
    elif failure == "inside_code":
        run = public_stage.code / "nested-run"
        run.mkdir(mode=0o700)
        monkeypatch.setenv("PERSONALITY_RUN_ROOT", str(run))
    else:
        monkeypatch.setattr(helper.stat, "S_IMODE", lambda mode: 0o755)
    with pytest.raises(helper.PrivateStageError):
        helper.run_private_compile()
    public_stage.compile.assert_not_called()
    public_stage.pin.assert_not_called()


@pytest.mark.parametrize("failure", ["source_in_code", "pin", "training_authority", "verification_mismatch", "compiler_payload"])
def test_source_refusals_and_payload_errors_are_sanitized(public_stage, monkeypatch, capsys, failure):
    token = "FICTITIOUS SOURCE SENTENCE MUST NOT REACH LOGS"
    if failure == "source_in_code":
        path = public_stage.code / "public-fixture.zip"
        path.write_bytes(b"public fixture")
        monkeypatch.setenv("PRIVATE_PACKAGE_PATH", str(path))
    elif failure == "pin":
        public_stage.pin.return_value = SimpleNamespace(archive_sha256="0" * 64)
    elif failure == "training_authority":
        public_stage.receipt["private_gradient_authorized"] = True
    elif failure == "verification_mismatch":
        public_stage.verify.side_effect = lambda output, **kwargs: {**public_stage.receipt, "active_record_count": 500}
    else:
        def raise_payload(*args, **kwargs):
            print(token)
            raise ValueError(token)
        public_stage.compile.side_effect = raise_payload
    assert helper.main(["--p2-compile-stage"]) == 3
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err
    path = public_stage.run / "compile_summary.json"
    parsed = json.loads(path.read_bytes())
    assert parsed["state"] == "FAILED_UNQUALIFIED"
    assert parsed["failure_reason"] == "private_stage_refused"
    assert token not in path.read_text()
    assert parsed["private_gradient_authorized"] is False and parsed["acceptance_authority"] is False
    if failure in {"source_in_code", "pin"}:
        public_stage.compile.assert_not_called()
    if failure in {"training_authority", "verification_mismatch", "compiler_payload"}:
        # Preserve partial output for review rather than deleting or replacing it.
        if failure != "compiler_payload":
            assert (public_stage.run / "compiled-substrate/public-fixture.jsonl").exists()


def test_exact_git_contract_uses_legacy_flags_and_no_output_leak(monkeypatch, tmp_path):
    runner = Mock(side_effect=[SimpleNamespace(stdout="a" * 40 + "\n"), SimpleNamespace(stdout="")])
    monkeypatch.setattr(helper.subprocess, "run", runner)
    helper._clean_code(tmp_path, "a" * 40)
    for call in runner.call_args_list:
        command = call.args[0]
        assert "-C" not in command
        assert f"--git-dir={tmp_path / '.git'}" in command
        assert f"--work-tree={tmp_path}" in command
        assert call.kwargs["capture_output"] is True
        assert call.kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
    runner.side_effect = [SimpleNamespace(stdout="c" * 40), SimpleNamespace(stdout="PUBLIC_UNTRACKED_FILENAME")]
    with pytest.raises(helper.PrivateStageError, match="exact clean"):
        helper._clean_code(tmp_path, "a" * 40)


def test_public_paths_must_be_bounded_and_summary_is_create_only(tmp_path):
    root = tmp_path / "owner-root"
    root.mkdir()
    public = root / "public-input.json"
    public.write_text("{}")
    assert helper._bounded_path(str(public), root, file=True) == public
    outside = tmp_path / "public-outside.json"
    outside.write_text("{}")
    with pytest.raises(helper.PrivateStageError):
        helper._bounded_path(str(outside), root, file=True)
    with pytest.raises(helper.PrivateStageError):
        helper._bounded_path("relative-public-path", root, file=True)
    receipt = root / "public-receipt.json"
    helper._write_summary(receipt, {"state": "FAILED_UNQUALIFIED"})
    prior = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        helper._write_summary(receipt, {"state": "COMPILED_UNQUALIFIED"})
    assert receipt.read_bytes() == prior


def test_transport_and_single_existing_container_stage_contract():
    raw = SBATCH.read_bytes()
    code = raw.decode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert code.count('exec "$UDOCKER" run') == 1
    assert "module load python/3.11.5" in code
    assert 'exec python3 -I "$@"' in code and 'exec python -I "$1" --p2-compile-stage' in code
    assert "--entrypoint=/bin/bash rayan-n0-base" in code
    assert "#SBATCH --partition=node" in code and "#SBATCH --mem=4G" in code
    assert "#SBATCH --time=00:15:00" in code
    for forbidden in ('"$UDOCKER" setup', '"$UDOCKER" pull', '"$UDOCKER" mode', "--map-current-user", "pip install", "--gres"):
        assert forbidden not in code
    assert 'realpath -e "$PRIVATE_PACKAGE_PATH"' not in code
    assert 'stat -c %a "$PRIVATE_PACKAGE_PATH"' not in code
    assert 'sha256sum "$PRIVATE_PACKAGE_PATH"' not in code
    for forbidden in ('realpath -e "$RAW_SOURCE"', 'stat -c %a "$RAW_SOURCE"', 'sha256sum "$RAW_SOURCE"'):
        assert forbidden not in code
    assert '--env="PERSONALITY_RAW_SOURCE_PATH=$PERSONALITY_RAW_SOURCE_PATH"' in code
    assert '--env="PERSONALITY_RAW_MEMBER_MAP_PATH=$PERSONALITY_RAW_MEMBER_MAP_PATH"' in code
    assert '--env="PERSONALITY_RAW_MEMBER_MAP_SHA256=$PERSONALITY_RAW_MEMBER_MAP_SHA256"' in code
    for forbidden in ('realpath -e "$MEMBER_MAP"', 'stat -c %a "$MEMBER_MAP"', 'sha256sum "$MEMBER_MAP"'):
        assert forbidden not in code
    assert 'export PERSONALITY_HOST_NETNS="$(readlink /proc/self/ns/net)"' in code
    assert '[[ ! -e "$PERSONALITY_RUN_ROOT" && ! -L "$PERSONALITY_RUN_ROOT" ]]' in code
