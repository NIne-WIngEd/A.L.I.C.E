"""Public synthetic ZIPs only; no real source, namespace or source admission.

Archive/member digests here attest tiny public fixture bytes, never private
payloads. Namespace tests mock a refusal before any path work; they do not
establish Magnolia/P2 isolation. Bash syntax checking is read-only/local.
"""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest

from src.alice_personality.n1 import compiler
from src.alice_personality.n1 import manifest_structure_audit as audit
from src.alice_personality.n1 import raw_inference_lineage as raw


REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/eipm/n1/audit_provenance_manifest_structure.py"
SBATCH = REPO / "scripts/eipm/n1/magnolia_audit_provenance_manifests.sbatch"
SPEC = importlib.util.spec_from_file_location("manifest_structure_entry_fixture", ENTRY)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def digest(data):
    return sha256(data).hexdigest()


def write_zip(path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return digest(path.read_bytes())


def fixture_inputs(tmp_path, *, curated_json=None, raw_json=None, nested_ledger=False,
                   raw_manifest_digest=None, extra_curated=None):
    curated_json = curated_json if curated_json is not None else json.dumps({
        "public_fixture_key": {"inputs": [{"digest": "SCALAR_DIGEST_VALUE_NOT_FOR_SCHEMA", "count": 12345}],
                               "enabled": True, "empty": None},
        "protected_key_can_be_a_fixture_filename.jsonl": "STRING_VALUE_NOT_FOR_SCHEMA"}).encode()
    raw_json = raw_json if raw_json is not None else json.dumps({
        "another_public_fixture_key": {"input_files": ["PUBLIC_FIXTURE_ID_NOT_FOR_SCHEMA"], "ratio": 0.625}}).encode()
    root = "curated"
    members = {f"{root}/{name}": b"PUBLIC FIXTURE PAYLOAD MUST NEVER BE OPENED"
               for name in [*compiler.ACTIVE_FILES.values(), compiler.UNKNOWN_FILE, compiler.ALTERNATIVE_FILE]}
    members[root + "/curation_manifest.json"] = curated_json
    if nested_ledger:
        members[root + "/nested/SHA256SUMS.txt"] = b"PUBLIC NESTED LEDGER MUST NEVER BE OPENED"
        members[root + "/nested/fixture.jsonl"] = b"PUBLIC NESTED PAYLOAD MUST NEVER BE OPENED"
    if extra_curated:
        members.update(extra_curated)
    # Root ledger deliberately need not flatten nested-ledger membership.
    declared = {name: payload for name, payload in members.items() if "/nested/" not in name}
    members[root + "/SHA256SUMS.txt"] = b"".join(
        f"{digest(payload)}  {name}\n".encode() for name, payload in sorted(declared.items()))
    curated = tmp_path / "curated-public-fixture.zip"
    pin = compiler.PackagePin(write_zip(curated, members), root,
                              {name: digest(payload) for name, payload in members.items()})
    raw_source_name = "raw/exact_reviewed_public_source.jsonl"
    raw_manifest_name, raw_ledger_name = "raw/generation_manifest.json", "raw/SHA256SUMS.txt"
    raw_members = {raw_source_name: b"PUBLIC PROPOSAL PAYLOAD MUST NEVER BE OPENED", raw_manifest_name: raw_json}
    raw_members[raw_ledger_name] = (f"{digest(raw_members[raw_source_name])}  {raw_source_name}\n"
        f"{raw_manifest_digest or digest(raw_json)}  {raw_manifest_name}\n").encode()
    raw_path = tmp_path / "raw-public-fixture.zip"
    raw_sha = write_zip(raw_path, raw_members)
    member_map = {"schema": "alice-personality-raw-inference-source-layout-review-v1",
        "state": "PROPOSED_UNQUALIFIED", "source_archive_sha256": raw_sha,
        "source_layout_receipt_file_sha256": "1" * 64, "source_layout_receipt_sha256": "2" * 64,
        "source_member_path": raw_source_name, "generation_manifest_path": raw_manifest_name,
        "checksum_ledger_path": raw_ledger_name, "selection_basis": "public mechanical fixture only",
        **{key: False for key in audit._FLAGS}}
    return {"curated_archive_path": curated, "curated_pin": pin, "raw_archive_path": raw_path,
            "expected_raw_archive_sha256": raw_sha, "raw_member_map": member_map}


def test_only_two_exact_manifests_and_two_ledgers_are_opened(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path)
    opened = []
    original = zipfile.ZipFile.open

    def observed(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", observed)
    protected, summary = audit.discover_manifest_structure(**inputs)
    assert opened == ["curated/SHA256SUMS.txt", "curated/curation_manifest.json",
                      "raw/SHA256SUMS.txt", "raw/generation_manifest.json"]
    assert summary["opened_identity_payload_member_count"] == 0
    assert summary["link_compared"] is False and summary["digest_search_performed"] is False
    assert all(summary[key] is False for key in audit._FLAGS)
    protected_bytes = raw._canonical(protected)
    public_bytes = raw._canonical(summary)
    assert b"protected_key_can_be_a_fixture_filename.jsonl" in protected_bytes
    assert b"protected_key_can_be_a_fixture_filename.jsonl" not in public_bytes
    for scalar in (b"SCALAR_DIGEST_VALUE_NOT_FOR_SCHEMA", b"STRING_VALUE_NOT_FOR_SCHEMA",
                   b"PUBLIC_FIXTURE_ID_NOT_FOR_SCHEMA", b"12345", b"0.625"):
        assert scalar not in protected_bytes and scalar not in public_bytes
    audit.verify_discovery_result(protected, summary)


def test_nested_checksum_ledger_not_opened_or_required_to_be_flattened(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path, nested_ledger=True)
    original = raw._read_member
    opened = []

    def observed(archive, member, limit):
        opened.append(member.filename)
        assert "/nested/" not in member.filename
        return original(archive, member, limit)

    monkeypatch.setattr(raw, "_read_member", observed)
    audit.discover_manifest_structure(**inputs)
    assert len(opened) == 4


@pytest.mark.parametrize("payload", [b'{"duplicate":1,"duplicate":2}', b'{"value":NaN}',
                                     b'{"value":1e999}', b'["not a manifest object"]'])
def test_invalid_json_is_fixed_failure_without_value_echo(tmp_path, payload):
    with pytest.raises(audit.ManifestStructureAuditError) as failure:
        audit.discover_manifest_structure(**fixture_inputs(tmp_path, raw_json=payload))
    assert str(failure.value) in {"audit_refused", "audit_metadata_refused"}
    assert "duplicate" not in str(failure.value) and "value" not in str(failure.value)


@pytest.mark.parametrize("flag", audit._FLAGS)
def test_map_never_grants_authority(tmp_path, flag):
    inputs = fixture_inputs(tmp_path)
    inputs["raw_member_map"][flag] = True
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_input_invalid"):
        audit.discover_manifest_structure(**inputs)


def test_wrong_archive_pin_refuses_before_member_open(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path)
    inputs["curated_pin"] = compiler.PackagePin("0" * 64, inputs["curated_pin"].package_root,
                                               inputs["curated_pin"].members_sha256)
    opened = Mock()
    monkeypatch.setattr(zipfile.ZipFile, "open", opened)
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_custody_refused"):
        audit.discover_manifest_structure(**inputs)
    opened.assert_not_called()


def test_exact_curated_inventory_is_required(tmp_path):
    inputs = fixture_inputs(tmp_path)
    previous = inputs["curated_pin"]
    omitted = dict(previous.members_sha256)
    # Keep the compiler's required file inventory but remove a new extra member.
    with zipfile.ZipFile(inputs["curated_archive_path"], "a") as archive:
        archive.writestr("curated/unpinned_fixture.txt", "public stand-in")
    inputs["curated_pin"] = compiler.PackagePin(digest(inputs["curated_archive_path"].read_bytes()),
                                               previous.package_root, omitted)
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_inventory_refused"):
        audit.discover_manifest_structure(**inputs)


def test_namespace_directory_alias_refused_without_payload_read(tmp_path):
    inputs = fixture_inputs(tmp_path, extra_curated={"curated/Case/a.txt": b"fixture",
                                                    "curated/case/b.txt": b"fixture"})
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_refused"):
        audit.discover_manifest_structure(**inputs)


def test_manifest_digest_must_match_selected_ledger(tmp_path):
    inputs = fixture_inputs(tmp_path, raw_manifest_digest="0" * 64)
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_metadata_refused"):
        audit.discover_manifest_structure(**inputs)


def test_crc_error_cannot_echo_protected_member_name(tmp_path):
    inputs = fixture_inputs(tmp_path)
    path = inputs["raw_archive_path"]
    source_name = inputs["raw_member_map"]["source_member_path"]
    manifest_name = inputs["raw_member_map"]["generation_manifest_path"]
    ledger_name = inputs["raw_member_map"]["checksum_ledger_path"]
    manifest = b'{"fixture":"UNIQUEPUBLICFIXTURECRC"}'
    proposal = b"public proposal fixture, never opened by audit"
    ledger = (f"{digest(proposal)}  {source_name}\n{digest(manifest)}  {manifest_name}\n").encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in ((source_name, proposal), (manifest_name, manifest), (ledger_name, ledger)):
            archive.writestr(name, payload)
    payload = path.read_bytes()
    assert payload.count(b"UNIQUEPUBLICFIXTURECRC") == 1
    path.write_bytes(payload.replace(b"UNIQUEPUBLICFIXTURECRC", b"uniquePUBLICFIXTURECRC"))
    inputs["expected_raw_archive_sha256"] = digest(path.read_bytes())
    inputs["raw_member_map"]["source_archive_sha256"] = inputs["expected_raw_archive_sha256"]
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_refused$") as failure:
        audit.discover_manifest_structure(**inputs)
    assert manifest_name not in str(failure.value)


def test_protected_selection_is_exact_not_basename_guess(tmp_path):
    inputs = fixture_inputs(tmp_path)
    inputs["raw_member_map"]["generation_manifest_path"] = "other/generation_manifest.json"
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_input_invalid"):
        audit.discover_manifest_structure(**inputs)


def test_changed_archive_during_later_manifest_read_fails_return(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path)
    original = raw._read_member

    def mutate(archive, member, limit):
        result = original(archive, member, limit)
        if member.filename == "raw/generation_manifest.json":
            with inputs["curated_archive_path"].open("ab") as stream:
                stream.write(b"public fixture mutation")
        return result

    monkeypatch.setattr(raw, "_read_member", mutate)
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_custody_refused"):
        audit.discover_manifest_structure(**inputs)


def test_changed_implementation_fails_return(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path)
    original = audit.implementation_hashes()
    changed = dict(original, **{"compiler.py": "0" * 64})
    monkeypatch.setattr(audit, "implementation_hashes", Mock(side_effect=[original, changed]))
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_code_changed"):
        audit.discover_manifest_structure(**inputs)


@pytest.mark.parametrize("constant, limit, value", [
    ("MAX_STRUCTURE_DEPTH", 1, {"a": {"b": 1}}),
    ("MAX_STRUCTURE_NODES", 2, {"a": [1, 2]}),
    ("MAX_STRUCTURE_KEY_BYTES", 2, {"long_fixture_key": 1}),
    ("MAX_STRUCTURE_BYTES", 10, {"a": 1}),
])
def test_structure_bounds_are_enforced(monkeypatch, constant, limit, value):
    monkeypatch.setattr(audit, constant, limit)
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_structure_bound"):
        audit.manifest_structure(value)


def test_fixed_summary_rejects_extra_keys_truthy_flags_and_scalar_payload(tmp_path):
    protected, summary = audit.discover_manifest_structure(**fixture_inputs(tmp_path))
    bad = dict(summary, PRIVATE_FIXTURE_KEY="PRIVATE FIXTURE VALUE")
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_metadata_refused"):
        audit.verify_discovery_result(protected, bad)
    bad = dict(summary, link_compared=0)
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_metadata_refused"):
        audit.verify_discovery_result(protected, bad)
    leaked = deepcopy(protected)
    leaked["curated"]["structure"]["fields"]["leaked"] = {"type": "string", "value": "PRIVATE FIXTURE VALUE"}
    changed = dict(summary, protected_structure_sha256=digest(raw._canonical(leaked)))
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_metadata_refused"):
        audit.verify_discovery_result(leaked, changed)


def test_persistent_same_byte_symlink_substitution_is_refused(tmp_path, monkeypatch):
    source, target = tmp_path / "map.json", tmp_path / "same.json"
    source.write_bytes(b"public fixture bytes")
    target.write_bytes(source.read_bytes())
    with pytest.raises((audit.ManifestStructureAuditError, raw.RawInferenceLineageError)):
        with audit._PinnedFile(source, digest(source.read_bytes()), 1024):
            if entry.sys.platform == "linux":
                source.unlink()
                source.symlink_to(target)
            else:
                # Windows refuses unlink of the open descriptor itself. Simulate
                # the linked-path observation while retaining actual same bytes.
                monkeypatch.setattr(raw, "_archive_path", Mock(side_effect=
                                    raw.RawInferenceLineageError("linked_archive_path")))


def test_pinned_map_rechecked_after_parse(tmp_path):
    path = tmp_path / "map.json"
    path.write_bytes(b"public fixture map")
    with pytest.raises(audit.ManifestStructureAuditError, match="audit_custody_refused"):
        with audit._PinnedFile(path, digest(path.read_bytes()), 1024) as pinned:
            assert pinned.bounded_bytes() == b"public fixture map"
            path.write_bytes(b"changed public fixture map")


def test_actual_guard_precedes_all_private_path_operations(monkeypatch):
    guard = Mock(side_effect=entry.namespace.PrivateStageError("public guard refusal"))
    bounded, clean = Mock(), Mock()
    monkeypatch.setattr(entry.namespace, "_p2_guard", guard)
    monkeypatch.setattr(entry.namespace, "_bounded_path", bounded)
    monkeypatch.setattr(entry.namespace, "_clean_code", clean)
    # Even invalid/missing env paths must not be examined before guard refusal.
    monkeypatch.delenv("COMPUTE_ROOT", raising=False)
    with pytest.raises(entry.namespace.PrivateStageError, match="public guard refusal"):
        entry.run_structure_audit()
    guard.assert_called_once()
    bounded.assert_not_called()
    clean.assert_not_called()


def test_entry_failure_logs_only_fixed_schema(monkeypatch, capsys):
    monkeypatch.setattr(entry, "run_structure_audit", Mock(side_effect=ValueError("PRIVATE FIXTURE PATH OR ID")))
    assert entry.main(["--p2-structure-stage"]) == 3
    result = capsys.readouterr()
    assert result.out == "" and "PRIVATE FIXTURE" not in result.err
    public = json.loads(result.err)
    assert public == {"state": "FAILED_UNQUALIFIED", "failure_reason": "manifest_structure_audit_refused",
                      "link_compared": False, "acceptance_authority": False, "private_gradient_authorized": False}
    assert audit.safe_reason(audit.ManifestStructureAuditError("PRIVATE FIXTURE ID")) == "audit_refused"


def test_create_only_protected_output_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "protected.json"
    original = entry.os.fstat
    # Actual0600 is a Linux protection property; Windows fixture mocks that check.
    if entry.sys.platform != "linux":
        monkeypatch.setattr(entry.os, "fstat", lambda fd: SimpleNamespace(st_mode=stat.S_IFREG | 0o600))
    entry._write_new(path, b"public fixture structure")
    with pytest.raises(FileExistsError):
        entry._write_new(path, b"replacement denied")
    assert path.read_bytes() == b"public fixture structure"
    if entry.sys.platform == "linux":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_final_summary_seal_and_file_hash_are_checked(tmp_path):
    protected, initial = audit.discover_manifest_structure(**fixture_inputs(tmp_path))
    payload = raw._canonical(protected)
    summary = dict(initial, raw_member_map_file_sha256="3" * 64, code_commit="4" * 40,
                   protected_structure_file_sha256=digest(payload + b"\n"),
                   entry_implementation_sha256={"entry.py": "5" * 64, "namespace.py": "6" * 64},
                   isolation={"host_uid": 1905, "host_gid": 100, "host_netns": "net:[100]",
                              "isolated_netns": "net:[200]", "interfaces": ["lo"],
                              "p2_effective_uid": 0, "p2_effective_gid": 0})
    summary["receipt_sha256"] = digest(raw._canonical(summary))
    entry._validate_final_summary(summary, initial, payload)
    for change in ({"receipt_sha256": "0" * 64}, {"protected_structure_file_sha256": "0" * 64},
                   {"PRIVATE_FIXTURE_KEY": "must not be public"}, {"isolation": {"host_uid": 0}}):
        with pytest.raises(entry.namespace.PrivateStageError):
            entry._validate_final_summary(dict(summary, **change), initial, payload)


def test_launcher_reuses_namespace_and_one_existing_p2_no_payload_work():
    text = SBATCH.read_text()
    assert "#SBATCH --mem=4G" in text and "#SBATCH --cpus-per-task=1" in text and "#SBATCH --time=00:15:00" in text
    assert '"$NAMESPACE_HELPER" -- /bin/bash' in text
    assert 'exec python -I "$1" --p2-structure-stage' in text
    assert "rayan-n0-base" in text and text.count('exec "$UDOCKER" run') == 1
    assert "rayan-provenance-manifests-${SLURM_JOB_ID}" in text
    for forbidden in ("--p2-compile-stage", "pip install", "udocker setup", "derive_raw_inference_registry", "compile_package"):
        assert forbidden not in text
    # Before actualnamespace/P2 route, private paths occur only in lexical checks/env.
    prefix = text.split('"$NAMESPACE_HELPER" -- /bin/bash')[0]
    for private in ("PRIVATE_PACKAGE_PATH", "PERSONALITY_RAW_SOURCE_PATH", "PERSONALITY_RAW_MEMBER_MAP_PATH"):
        assert f'realpath -e "${private}"' not in prefix
        assert f'stat -c %U "${private}"' not in prefix
    bash = shutil.which("bash")
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        bash = str(git_bash)
    if bash:
        result = subprocess.run([bash, "-n", str(SBATCH)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
