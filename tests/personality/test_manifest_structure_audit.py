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
COMPARISON_SBATCH = REPO / "scripts/eipm/n1/magnolia_compare_declared_source_hashes.sbatch"
FORMAT_SBATCH = REPO / "scripts/eipm/n1/magnolia_diagnose_declared_source_hash_formats.sbatch"
BYTES_SBATCH = REPO / "scripts/eipm/n1/magnolia_compare_declared_source_hash_bytes.sbatch"
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
    with pytest.raises(entry.SafeAuditFailure) as failure:
        entry.run_structure_audit()
    assert failure.value.stage == "namespace_guard"
    assert failure.value.category == "boundary_or_schema_refused"
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
                      "failure_stage": "entry_invocation", "failure_category": "unclassified_refusal",
                      "link_compared": False, "acceptance_authority": False, "private_gradient_authorized": False,
                      "historical_authority_granted": False, "training_authorized": False,
                      "source_admission": False, "original_source_independently_verified": False,
                      "legacy_namespace_proven": False}
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


def fixture_comparison(tmp_path, *, matches=0, manifest_change=None):
    values = {f"PUBLIC_FIXTURE_LABEL_{number:02d}": f"{number + 10:064x}" for number in range(30)}
    for number in range(matches):
        values[f"PUBLIC_FIXTURE_LABEL_{number:02d}"] = audit.LEGACY_RESERVE_SHA256
    manifest = {"source_hashes": values, "source_package": "PUBLIC_PACKAGE_VALUE_MUST_NOT_BE_LOGGED",
                # A matching digest outside the exact source_hashes path is ignored.
                "notes": {"unrelated_output_digest": audit.LEGACY_RESERVE_SHA256}}
    if manifest_change:
        manifest_change(manifest)
    inputs = fixture_inputs(tmp_path, raw_json=raw._canonical(manifest))
    protected, discovery = audit.discover_manifest_structure(**inputs)
    path = tmp_path / "reviewed-public-fixture.protected.json"
    path.write_bytes(raw._canonical(protected) + b"\n")
    return dict(inputs, reviewed_structure_path=path,
                expected_reviewed_structure_file_sha256=digest(path.read_bytes()),
                expected_curated_manifest_sha256=discovery["curated_manifest_sha256"],
                expected_raw_generation_manifest_sha256=discovery["raw_generation_manifest_sha256"])


@pytest.mark.parametrize("matches", [0, 1, 2])
def test_targeted_declaration_count_without_label_or_original_authority(tmp_path, monkeypatch, matches):
    inputs = fixture_comparison(tmp_path, matches=matches)
    before = inputs["reviewed_structure_path"].read_bytes()
    opened = []
    original = raw._read_member

    def observed(archive, member, limit):
        opened.append(member.filename)
        return original(archive, member, limit)

    monkeypatch.setattr(raw, "_read_member", observed)
    result = audit.compare_declared_source_hashes(**inputs)
    assert result["reserve_digest_match_count"] == matches
    assert result["generator_declares_matching_digest"] is (matches > 0)
    assert result["declared_source_hash_count"] == 30
    assert result["link_compared"] is True
    assert all(result[key] is False for key in (*audit._FLAGS, "original_source_independently_verified",
                                               "legacy_namespace_proven", "source_admission", "digest_search_performed"))
    assert len(opened) == 4 and not any(name.endswith(".jsonl") for name in opened)
    public = raw._canonical(result)
    assert b"PUBLIC_FIXTURE_LABEL" not in public and b"PUBLIC_PACKAGE_VALUE" not in public
    assert inputs["reviewed_structure_path"].read_bytes() == before
    audit.verify_comparison_result(result)


@pytest.mark.parametrize("invalid", [True, 42, "g" * 64, "A" * 64, "0" * 63, {"digest": "0" * 64}])
def test_all30_declared_values_must_validate_before_any_match_result(tmp_path, invalid):
    inputs = fixture_comparison(tmp_path, matches=1, manifest_change=lambda value:
        value["source_hashes"].__setitem__("PUBLIC_FIXTURE_LABEL_29", invalid))
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_declared_digest_format_refused$"):
        audit.compare_declared_source_hashes(**inputs)


def test_declared_map_requires_exact30_and_package_string(tmp_path):
    for number, change in enumerate((lambda value: value["source_hashes"].pop("PUBLIC_FIXTURE_LABEL_29"),
                                    lambda value: value.__setitem__("source_package", True),
                                    lambda value: value.__setitem__("source_hashes", ["0" * 64] * 30))):
        directory = tmp_path / str(number)
        directory.mkdir()
        inputs = fixture_comparison(directory, manifest_change=change)
        with pytest.raises(audit.ManifestStructureAuditError, match="^audit_declared_map_shape_refused$"):
            audit.compare_declared_source_hashes(**inputs)


def test_reviewed_structure_exact_typed_bytes_reject_false_zero_and_key_drift(tmp_path):
    inputs = fixture_comparison(tmp_path)
    path = inputs["reviewed_structure_path"]
    reviewed = json.loads(path.read_bytes())
    for changed in (dict(reviewed, link_compared=0), deepcopy(reviewed)):
        if changed.get("link_compared") is False:
            fields = changed["raw"]["structure"]["fields"]["source_hashes"]["fields"]
            fields["DIFFERENT_PUBLIC_FIXTURE_LABEL"] = fields.pop("PUBLIC_FIXTURE_LABEL_29")
            assert len(fields) == 30
        path.write_bytes(raw._canonical(changed) + b"\n")
        inputs["expected_reviewed_structure_file_sha256"] = digest(path.read_bytes())
        with pytest.raises(audit.ManifestStructureAuditError, match="^audit_reviewed_structure_refused$"):
            audit.compare_declared_source_hashes(**inputs)


def test_reviewed_structure_and_actual_manifest_pins_are_both_required(tmp_path, monkeypatch):
    inputs = fixture_comparison(tmp_path)
    opened = Mock()
    monkeypatch.setattr(zipfile.ZipFile, "open", opened)
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_custody_refused$"):
        audit.compare_declared_source_hashes(**dict(inputs, expected_reviewed_structure_file_sha256="0" * 64))
    opened.assert_not_called()
    monkeypatch.undo()
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_reviewed_manifest_pins_refused$"):
        audit.compare_declared_source_hashes(**dict(inputs, expected_raw_generation_manifest_sha256="0" * 64))


def test_comparison_observer_source_mutation_cannot_return_result(tmp_path, monkeypatch):
    inputs = fixture_comparison(tmp_path)
    original = audit._declared_matches

    def changed(manifest):
        result = original(manifest)
        with inputs["raw_archive_path"].open("ab") as stream:
            stream.write(b"public fixture mutation after targeted comparison")
        return result

    monkeypatch.setattr(audit, "_declared_matches", changed)
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_custody_refused$"):
        audit.compare_declared_source_hashes(**inputs)


def test_comparison_result_rejects_inconsistent_counts_booltypes_and_authority(tmp_path):
    result = audit.compare_declared_source_hashes(**fixture_comparison(tmp_path, matches=1))
    for change in ({"reserve_digest_match_count": 31}, {"reserve_digest_match_count": True},
                   {"generator_declares_matching_digest": False}, {"source_admission": 0},
                   {"original_source_independently_verified": True}, {"PRIVATE_FIXTURE_LABEL": "forbidden"}):
        with pytest.raises(audit.ManifestStructureAuditError, match="^audit_metadata_refused$"):
            audit.verify_comparison_result(dict(result, **change))


def test_comparison_mode_guard_precedes_private_inputs_and_failure_is_fixed(monkeypatch, capsys):
    guard, bounded = Mock(side_effect=entry.namespace.PrivateStageError("PUBLIC GUARD REFUSAL")), Mock()
    monkeypatch.setattr(entry.namespace, "_p2_guard", guard)
    monkeypatch.setattr(entry.namespace, "_bounded_path", bounded)
    with pytest.raises(entry.namespace.PrivateStageError):
        entry.run_structure_audit(compare_source_hashes=True)
    bounded.assert_not_called()
    monkeypatch.setattr(entry, "run_structure_audit", Mock(side_effect=ValueError("PRIVATE FIXTURE LABEL")))
    assert entry.main(["--p2-source-hash-comparison-stage"]) == 3
    output = capsys.readouterr()
    assert output.out == "" and "PRIVATE FIXTURE" not in output.err


def test_public_target_pin_and_new_launcher_contract():
    pin = compiler.curated_frontier_v2_pin()  # PUBLIC pin file only; no ZIP open.
    assert pin.members_sha256[pin.package_root + "/" + audit.LEGACY_RESERVE_PUBLIC_MEMBER] == audit.LEGACY_RESERVE_SHA256
    text = COMPARISON_SBATCH.read_text()
    assert "rayan-source-hash-comparison-${SLURM_JOB_ID}" in text
    assert 'exec python -I "$1" --p2-source-hash-comparison-stage' in text
    assert '--env="PERSONALITY_REVIEWED_STRUCTURE_PATH=$PERSONALITY_REVIEWED_STRUCTURE_PATH"' in text
    assert "#SBATCH --mem=4G" in text and "#SBATCH --cpus-per-task=1" in text and "#SBATCH --time=00:15:00" in text
    assert text.count('exec "$UDOCKER" run') == 1 and "rayan-n0-base" in text
    prefix = text.split('"$NAMESPACE_HELPER" -- /bin/bash')[0]
    assert 'realpath -e "$PERSONALITY_REVIEWED_STRUCTURE_PATH"' not in prefix
    bash = str(Path("C:/Program Files/Git/bin/bash.exe")) if Path("C:/Program Files/Git/bin/bash.exe").is_file() else shutil.which("bash")
    if bash:
        checked = subprocess.run([bash, "-n", str(COMPARISON_SBATCH)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr


def test_guard_passed_missing_boundary_env_is_reported_before_any_source_path(monkeypatch, capsys):
    monkeypatch.setattr(entry.namespace, "_p2_guard", Mock(return_value={}))
    monkeypatch.delenv("COMPUTE_ROOT", raising=False)
    bounded, clean = Mock(), Mock()
    monkeypatch.setattr(entry.namespace, "_bounded_path", bounded)
    monkeypatch.setattr(entry.namespace, "_clean_code", clean)
    assert entry.main(["--p2-source-hash-comparison-stage"]) == 3
    output = capsys.readouterr()
    value = json.loads(output.err)
    assert value["failure_stage"] == "compute_boundary"
    assert value["failure_category"] == "required_field_absent"
    assert "COMPUTE_ROOT" not in output.err and output.out == ""
    bounded.assert_not_called()
    clean.assert_not_called()


@pytest.mark.parametrize("exception, category", [
    (PermissionError(13, "PRIVATE FIXTURE MESSAGE", "/PRIVATE-FIXTURE-PATH"), "permission_refused"),
    (FileNotFoundError(2, "PRIVATE FIXTURE MESSAGE", "/PRIVATE-FIXTURE-PATH"), "required_input_absent"),
    (FileExistsError(17, "PRIVATE FIXTURE MESSAGE", "/PRIVATE-FIXTURE-PATH"), "create_only_conflict"),
    (KeyError("PRIVATE-FIXTURE-KEY"), "required_field_absent"),
    (ModuleNotFoundError("PRIVATE-FIXTURE-MODULE"), "public_dependency_unavailable"),
    (ValueError("audit_custody_refused"), "unclassified_refusal"),
    (audit.ManifestStructureAuditError("audit_custody_refused"), "input_custody_refused"),
    (audit.ManifestStructureAuditError("audit_metadata_refused"), "metadata_or_review_refused"),
    (audit.ManifestStructureAuditError("audit_reviewed_structure_refused"), "reviewed_structure_refused"),
    (audit.ManifestStructureAuditError("audit_reviewed_manifest_pins_refused"), "reviewed_manifest_pins_refused"),
    (audit.ManifestStructureAuditError("audit_declared_map_shape_refused"), "declared_map_shape_refused"),
    (audit.ManifestStructureAuditError("audit_declared_digest_format_refused"), "declared_digest_format_refused"),
    (ValueError("audit_declared_digest_format_refused"), "unclassified_refusal"),
    (audit.ManifestStructureAuditError("PRIVATE-FIXTURE-KEY"), "unclassified_refusal"),
])
def test_fixed_failure_stage_category_never_echoes_errors_or_forges_module_reason(monkeypatch, capsys, exception, category):
    def fails(*, compare_source_hashes, progress):
        progress.mark("declared_hash_comparison")
        raise exception

    monkeypatch.setattr(entry, "_run_structure_audit", fails)
    assert entry.main(["--p2-source-hash-comparison-stage"]) == 3
    output = capsys.readouterr()
    assert output.out == "" and "PRIVATE" not in output.err and "Traceback" not in output.err
    value = json.loads(output.err)
    assert value["failure_stage"] == "declared_hash_comparison"
    assert value["failure_category"] == category
    assert set(value) == {"state", "failure_reason", "failure_stage", "failure_category", "link_compared",
                         "acceptance_authority", "private_gradient_authorized", "historical_authority_granted",
                         "training_authorized", "source_admission", "original_source_independently_verified",
                         "legacy_namespace_proven"}
    assert all(value[name] is False for name in ("link_compared", *audit._FLAGS, "source_admission",
                                                "original_source_independently_verified", "legacy_namespace_proven"))


def test_failure_does_not_stringify_exception_or_expose_a_traceback(monkeypatch, capsys):
    class PoisonedError(ValueError):
        def __str__(self):
            raise AssertionError("exception stringification must never occur")

    def fails(*, compare_source_hashes, progress):
        progress.mark("member_map_schema")
        raise PoisonedError("PRIVATE FIXTURE PAYLOAD")

    monkeypatch.setattr(entry, "_run_structure_audit", fails)
    with pytest.raises(entry.SafeAuditFailure) as caught:
        entry.run_structure_audit(compare_source_hashes=True)
    assert str(caught.value) == "manifest_structure_audit_refused"
    assert caught.value.__suppress_context__ is True and caught.value.__cause__ is None
    assert entry.main(["--p2-source-hash-comparison-stage"]) == 3
    output = capsys.readouterr()
    assert output.out == "" and "PRIVATE" not in output.err and "PoisonedError" not in output.err
    assert json.loads(output.err)["failure_stage"] == "member_map_schema"


def test_forged_or_missing_failure_fields_collapse_to_static_defaults(monkeypatch, capsys):
    failure = entry.SafeAuditFailure("PRIVATE-FIXTURE-STAGE", "PRIVATE-FIXTURE-CATEGORY")
    failure.stage = "/PRIVATE-FIXTURE-PATH"
    del failure.category
    monkeypatch.setattr(entry, "run_structure_audit", Mock(side_effect=failure))
    assert entry.main(["--p2-structure-stage"]) == 3
    output = capsys.readouterr()
    value = json.loads(output.err)
    assert value["failure_stage"] == "entry_invocation" and value["failure_category"] == "unclassified_refusal"
    assert "PRIVATE" not in output.err and output.out == ""


def test_real_fixture_custody_failure_is_reported_and_cannot_publish(tmp_path, monkeypatch, capsys):
    path = tmp_path / "public-fixture-map.json"
    path.write_bytes(b"public fixture initial bytes")
    original = digest(path.read_bytes())
    publish = Mock()

    def fails(*, compare_source_hashes, progress):
        with audit._PinnedFile(path, original, 1024) as pinned:
            path.write_bytes(b"changed public fixture bytes")
            progress.mark("member_map_recheck")
            pinned.verify()
        publish()

    monkeypatch.setattr(entry, "_run_structure_audit", fails)
    assert entry.main(["--p2-source-hash-comparison-stage"]) == 3
    value = json.loads(capsys.readouterr().err)
    assert value["failure_stage"] == "member_map_recheck"
    assert value["failure_category"] == "input_custody_refused"
    publish.assert_not_called()


def test_create_only_failure_stage_preserves_existing_evidence(tmp_path, monkeypatch, capsys):
    path = tmp_path / "public-fixture-old-evidence.json"
    path.write_bytes(b"old public fixture evidence")

    def fails(*, compare_source_hashes, progress):
        progress.mark("public_summary_write")
        entry._write_new(path, b"must not replace old evidence")

    monkeypatch.setattr(entry, "_run_structure_audit", fails)
    assert entry.main(["--p2-structure-stage"]) == 3
    value = json.loads(capsys.readouterr().err)
    assert value["failure_stage"] == "public_summary_write" and value["failure_category"] == "create_only_conflict"
    assert path.read_bytes() == b"old public fixture evidence"


@pytest.mark.parametrize("case, category", [
    ("reviewed_structure", "reviewed_structure_refused"),
    ("reviewed_encoding", "reviewed_structure_refused"),
    ("curated_manifest_pin", "reviewed_manifest_pins_refused"),
    ("raw_manifest_pin", "reviewed_manifest_pins_refused"),
    ("map_shape", "declared_map_shape_refused"),
    ("package_shape", "declared_map_shape_refused"),
    ("digest_format", "declared_digest_format_refused"),
])
def test_actual_comparison_refusals_have_fixed_subcategories_after_fixture_guard(
        tmp_path, monkeypatch, capsys, case, category):
    # The guard is mocked; this is not evidence of actual Magnolia isolation.
    changes = {
        "map_shape": lambda value: value["source_hashes"].pop("PUBLIC_FIXTURE_LABEL_29"),
        "package_shape": lambda value: value.__setitem__("source_package", False),
        "digest_format": lambda value: value["source_hashes"].__setitem__("PUBLIC_FIXTURE_LABEL_29", "A" * 64),
    }
    inputs = fixture_comparison(tmp_path, matches=1, manifest_change=changes.get(case))
    reviewed_path = inputs["reviewed_structure_path"]
    if case in ("reviewed_structure", "reviewed_encoding"):
        reviewed = json.loads(reviewed_path.read_bytes())
        if case == "reviewed_structure":
            reviewed["link_compared"] = 0
            reviewed_bytes = raw._canonical(reviewed) + b"\n"
        else:
            reviewed_bytes = json.dumps(reviewed, indent=2).encode() + b"\n"
        reviewed_path.write_bytes(reviewed_bytes)
        inputs["expected_reviewed_structure_file_sha256"] = digest(reviewed_bytes)
    if case in ("curated_manifest_pin", "raw_manifest_pin"):
        field = "expected_curated_manifest_sha256" if case == "curated_manifest_pin" else "expected_raw_generation_manifest_sha256"
        inputs[field] = "0" * 64
    unchanged_review = reviewed_path.read_bytes()
    guard, publish = Mock(return_value={}), Mock()
    opened = []
    read_member = raw._read_member

    def observed(archive, member, limit):
        opened.append(member.filename)
        return read_member(archive, member, limit)

    def fixture_stage(*, compare_source_hashes, progress):
        progress.mark("namespace_guard")
        entry.namespace._p2_guard()
        progress.mark("declared_hash_comparison")
        result = audit.compare_declared_source_hashes(**inputs)
        publish(result)
        return result

    monkeypatch.setattr(entry.namespace, "_p2_guard", guard)
    monkeypatch.setattr(entry, "_run_structure_audit", fixture_stage)
    monkeypatch.setattr(raw, "_read_member", observed)
    assert entry.main(["--p2-source-hash-comparison-stage"]) == 3
    output = capsys.readouterr()
    value = json.loads(output.err)
    assert output.out == "" and "PUBLIC_FIXTURE_LABEL" not in output.err and "PUBLIC_PACKAGE_VALUE" not in output.err
    assert str(reviewed_path) not in output.err and "Traceback" not in output.err
    assert value["failure_stage"] == "declared_hash_comparison" and value["failure_category"] == category
    assert all(value[name] is False for name in ("link_compared", *audit._FLAGS, "source_admission",
                                                "original_source_independently_verified", "legacy_namespace_proven"))
    guard.assert_called_once_with()
    publish.assert_not_called()
    assert len(opened) == 4 and not any(name.endswith(".jsonl") for name in opened)
    assert reviewed_path.read_bytes() == unchanged_review


def test_reporting_refinement_preserves_original_structure_first_refusal_order(tmp_path):
    inputs = fixture_comparison(tmp_path, manifest_change=lambda value:
        value["source_hashes"].__setitem__("PUBLIC_FIXTURE_LABEL_29", "A" * 64))
    path = inputs["reviewed_structure_path"]
    reviewed = json.loads(path.read_bytes())
    reviewed["link_compared"] = 0
    path.write_bytes(raw._canonical(reviewed) + b"\n")
    inputs["expected_reviewed_structure_file_sha256"] = digest(path.read_bytes())
    inputs["expected_raw_generation_manifest_sha256"] = "0" * 64
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_reviewed_structure_refused$"):
        audit.compare_declared_source_hashes(**inputs)


def test_format_only_counts_all30_without_digest_matching_normalization_or_leak(tmp_path, monkeypatch):
    def changes(value):
        entries = value["source_hashes"]
        entries["PUBLIC_FIXTURE_LABEL_26"] = "a" * 63 + "F"
        entries["PUBLIC_FIXTURE_LABEL_27"] = ""
        entries["PUBLIC_FIXTURE_LABEL_28"] = "PRIVATE_FIXTURE_NONHASH_VALUE"
        entries["PUBLIC_FIXTURE_LABEL_29"] = "  " + "0" * 64
    inputs = fixture_comparison(tmp_path, matches=1, manifest_change=changes)
    originals = {name: inputs[name].read_bytes() for name in ("curated_archive_path", "raw_archive_path", "reviewed_structure_path")}
    opened, read_member = [], raw._read_member
    def observed(archive, member, limit):
        opened.append(member.filename)
        return read_member(archive, member, limit)
    monkeypatch.setattr(raw, "_read_member", observed)
    strict_matcher = audit._declared_matches
    matcher = Mock(side_effect=AssertionError("format diagnostic must not perform matching"))
    monkeypatch.setattr(audit, "_declared_matches", matcher)
    result = audit.diagnose_declared_source_hash_formats(**inputs)
    assert (result["lowercase_64hex_count"], result["uppercase_64hex_count"], result["empty_string_count"],
            result["other_nonempty_string_count"], result["declared_source_hash_count"]) == (26, 1, 1, 2, 30)
    assert result["schema"] == audit.FORMAT_SCHEMA and result["state"] == audit.FORMAT_STATE
    assert all(result[name] is False for name in (*audit._FLAGS, "link_compared", "digest_comparison_performed",
        "hash_normalization_performed", "digest_search_performed", "source_admission", "original_source_independently_verified",
        "legacy_namespace_proven", "reference_registry_written", "model_training_performed"))
    public = raw._canonical(result)
    assert all(text not in public for text in (b"PUBLIC_FIXTURE_LABEL", b"PRIVATE_FIXTURE_NONHASH_VALUE",
        b"PUBLIC_PACKAGE_VALUE", audit.LEGACY_RESERVE_SHA256.encode(), b"reserve_digest_match_count"))
    assert len(opened) == 4 and not any(name.endswith(".jsonl") for name in opened)
    matcher.assert_not_called()
    for name, before in originals.items():
        assert inputs[name].read_bytes() == before
    audit.verify_format_result(result)
    monkeypatch.setattr(audit, "_declared_matches", strict_matcher)
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_declared_digest_format_refused$"):
        audit.compare_declared_source_hashes(**inputs)  # The original strict route still refuses.


@pytest.mark.parametrize("target", ["raw_archive_path", "reviewed_structure_path"])
def test_format_diagnostic_cannot_return_after_held_input_mutation(tmp_path, monkeypatch, target):
    inputs = fixture_comparison(tmp_path)
    original = audit._declared_format_counts
    def mutate(value):
        counts = original(value)
        with inputs[target].open("ab") as stream:
            stream.write(b"PUBLIC_FIXTURE_MUTATION_AFTER_COUNT")
        return counts
    monkeypatch.setattr(audit, "_declared_format_counts", mutate)
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_custody_refused$"):
        audit.diagnose_declared_source_hash_formats(**inputs)


@pytest.mark.parametrize("change", [
    {"lowercase_64hex_count": True}, {"empty_string_count": -1}, {"other_nonempty_string_count": 31},
    {"lowercase_64hex_count": 29}, {"link_compared": True}, {"hash_normalization_performed": True},
    {"source_admission": True}, {"digest_comparison_performed": True}, {"PRIVATE_FIXTURE_LABEL": "forbidden"},
])
def test_format_result_requires_fixed_schema_exact_counts_and_false_authority(tmp_path, change):
    result = audit.diagnose_declared_source_hash_formats(**fixture_comparison(tmp_path))
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_metadata_refused$"):
        audit.verify_format_result(dict(result, **change))


def test_format_mode_guard_precedes_inputs_and_refusal_cannot_echo_private_values(monkeypatch, capsys):
    guard = Mock(side_effect=entry.namespace.PrivateStageError("PRIVATE_FIXTURE_PATH_OR_VALUE"))
    bounded = Mock()
    monkeypatch.setattr(entry.namespace, "_p2_guard", guard)
    monkeypatch.setattr(entry.namespace, "_bounded_path", bounded)
    assert entry.main(["--p2-source-hash-format-stage"]) == 3
    output = capsys.readouterr()
    failure = json.loads(output.err)
    assert failure["failure_stage"] == "namespace_guard" and "PRIVATE" not in output.err and output.out == ""
    assert failure["link_compared"] is False and failure["source_admission"] is False
    bounded.assert_not_called()


def test_format_mode_has_separate_fresh_one_stage_launcher_and_fixed_public_fields(tmp_path, monkeypatch, capsys):
    result = audit.diagnose_declared_source_hash_formats(**fixture_comparison(tmp_path))
    monkeypatch.setattr(entry, "run_structure_audit", Mock(return_value=dict(result, receipt_sha256="1" * 64)))
    assert entry.main(["--p2-source-hash-format-stage"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["link_compared"] is False and value["lowercase_64hex_count"] == 30
    assert "reserve_digest_match_count" not in value and "generator_declares_matching_digest" not in value
    entry.run_structure_audit.assert_called_once_with(diagnose_hash_formats=True)
    text = FORMAT_SBATCH.read_text()
    assert "rayan-source-hash-formats-${SLURM_JOB_ID}" in text and "#SBATCH --job-name=rayan-source-hash-formats" in text
    assert "#SBATCH --mem=4G" in text and "#SBATCH --cpus-per-task=1" in text and "#SBATCH --time=00:15:00" in text
    assert text.count('exec "$UDOCKER" run') == 1 and "rayan-n0-base" in text
    assert '"$NAMESPACE_HELPER" -- /bin/bash' in text and 'exec python -I "$1" --p2-source-hash-format-stage' in text
    prefix = text.split('"$NAMESPACE_HELPER" -- /bin/bash')[0]
    for field in ("PRIVATE_PACKAGE_PATH", "PERSONALITY_RAW_SOURCE_PATH", "PERSONALITY_RAW_MEMBER_MAP_PATH", "PERSONALITY_REVIEWED_STRUCTURE_PATH"):
        assert f'realpath -e "${field}"' not in prefix and f'stat -c %U "${field}"' not in prefix
    for forbidden in ("pip install", "udocker setup", "compile_package", "derive_raw_inference_registry", "--p2-source-hash-comparison-stage"):
        assert forbidden not in text
    bash = str(Path("C:/Program Files/Git/bin/bash.exe")) if Path("C:/Program Files/Git/bin/bash.exe").is_file() else shutil.which("bash")
    if bash:
        checked = subprocess.run([bash, "-n", str(FORMAT_SBATCH)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr


@pytest.mark.parametrize("case", ["lower", "upper", "mixed"])
@pytest.mark.parametrize("matches", [0, 2])
def test_v2_exact_digest_bytes_equivalence_does_not_mutate_source_or_v1(tmp_path, monkeypatch, case, matches):
    def change(value):
        for key, text in tuple(value["source_hashes"].items()):
            value["source_hashes"][key] = text if case == "lower" else text.upper() if case == "upper" else \
                "".join(char.upper() if index % 2 else char for index, char in enumerate(text))
    inputs = fixture_comparison(tmp_path, matches=matches, manifest_change=change)
    before = {key: inputs[key].read_bytes() for key in ("curated_archive_path", "raw_archive_path", "reviewed_structure_path")}
    opened, read_member = [], raw._read_member
    def observed(archive, member, limit):
        opened.append(member.filename)
        return read_member(archive, member, limit)
    monkeypatch.setattr(raw, "_read_member", observed)
    result = audit.compare_declared_source_hash_bytes(**inputs)
    assert result["schema"] == audit.COMPARISON_BYTES_SCHEMA and result["state"] == audit.COMPARISON_BYTES_STATE
    assert result["reserve_digest_match_count"] == matches and result["generator_declares_matching_digest"] is (matches > 0)
    assert result["comparison_representation"] == "decoded_32_byte_sha256_from_exact_64_ascii_hex"
    assert result["encoding_policy_sha256"] == audit.COMPARISON_BYTES_POLICY_SHA256
    assert result["decoded_digest_byte_count"] == 32 and result["declared_source_hash_count"] == 30
    assert all(result[key] is False for key in (*audit._FLAGS, "source_admission", "original_source_independently_verified",
        "legacy_namespace_proven", "digest_text_normalization_performed", "source_text_mutated"))
    assert len(opened) == 4 and not any(name.endswith(".jsonl") for name in opened)
    serialized = raw._canonical(result)
    assert b"PUBLIC_FIXTURE_LABEL" not in serialized and b"PUBLIC_PACKAGE_VALUE" not in serialized
    for key, payload in before.items():
        assert inputs[key].read_bytes() == payload
    audit.verify_bytes_comparison_result(result)
    if case == "lower":
        v1 = audit.compare_declared_source_hashes(**inputs)
        assert v1["schema"] == audit.COMPARISON_SCHEMA and v1["reserve_digest_match_count"] == matches
        assert "comparison_representation" not in v1
    else:
        with pytest.raises(audit.ManifestStructureAuditError, match="^audit_declared_digest_format_refused$"):
            audit.compare_declared_source_hashes(**inputs)


@pytest.mark.parametrize("invalid", [
    " " + "a" * 64, "a" * 64 + " ", "a" * 64 + "\n", "a" * 31 + "\t" + "a" * 32,
    "a" * 63, "a" * 65, "g" * 64, "Ａ" * 64, "é" + "a" * 63, "a" * 63 + "\x00", "", True,
])
def test_v2_all30_exact_ascii_formats_validate_before_any_decoder_or_equality(tmp_path, monkeypatch, invalid):
    inputs = fixture_comparison(tmp_path, matches=1, manifest_change=lambda value:
        value["source_hashes"].__setitem__("PUBLIC_FIXTURE_LABEL_29", invalid))
    decoder = Mock(side_effect=AssertionError("invalid map must refuse before decoding"))
    monkeypatch.setattr(audit.binascii, "unhexlify", decoder)
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_declared_digest_format_refused$"):
        audit.compare_declared_source_hash_bytes(**inputs)
    decoder.assert_not_called()


@pytest.mark.parametrize("target", ["curated_archive_path", "raw_archive_path", "reviewed_structure_path"])
def test_v2_source_and_reviewed_descriptors_rechecked_after_byte_comparison(tmp_path, monkeypatch, target):
    inputs = fixture_comparison(tmp_path, matches=1)
    original = audit._declared_byte_matches
    def mutate(value):
        count = original(value)
        with inputs[target].open("ab") as stream:
            stream.write(b"PUBLIC_FIXTURE_MUTATION_AFTER_BYTE_COMPARISON")
        return count
    monkeypatch.setattr(audit, "_declared_byte_matches", mutate)
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_custody_refused$"):
        audit.compare_declared_source_hash_bytes(**inputs)


@pytest.mark.parametrize("case, reason", [
    ("reviewed_file_pin", "audit_custody_refused"), ("reviewed_false_zero", "audit_reviewed_structure_refused"),
    ("curated_manifest_pin", "audit_reviewed_manifest_pins_refused"),
    ("raw_manifest_pin", "audit_reviewed_manifest_pins_refused"),
])
def test_v2_review_custody_and_exact_canonical_structure_precede_digest_decoding(tmp_path, monkeypatch, case, reason):
    inputs = fixture_comparison(tmp_path, matches=1)
    if case == "reviewed_file_pin":
        inputs["expected_reviewed_structure_file_sha256"] = "0" * 64
    elif case == "reviewed_false_zero":
        path = inputs["reviewed_structure_path"]
        reviewed = json.loads(path.read_bytes())
        reviewed["link_compared"] = 0
        path.write_bytes(raw._canonical(reviewed) + b"\n")
        inputs["expected_reviewed_structure_file_sha256"] = digest(path.read_bytes())
    else:
        key = "expected_curated_manifest_sha256" if case == "curated_manifest_pin" else "expected_raw_generation_manifest_sha256"
        inputs[key] = "0" * 64
    matcher = Mock(side_effect=AssertionError("byte comparison must follow complete reviewed custody"))
    monkeypatch.setattr(audit, "_declared_byte_matches", matcher)
    with pytest.raises(audit.ManifestStructureAuditError, match=f"^{reason}$"):
        audit.compare_declared_source_hash_bytes(**inputs)
    matcher.assert_not_called()


def test_v2_implementation_recheck_refuses_change_after_byte_comparison(tmp_path, monkeypatch):
    inputs = fixture_comparison(tmp_path)
    original = audit._declared_byte_matches
    changed = dict(audit.implementation_hashes(), **{"compiler.py": "0" * 64})
    def mutate(value):
        count = original(value)
        monkeypatch.setattr(audit, "implementation_hashes", Mock(return_value=changed))
        return count
    monkeypatch.setattr(audit, "_declared_byte_matches", mutate)
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_code_changed$"):
        audit.compare_declared_source_hash_bytes(**inputs)


@pytest.mark.parametrize("change", [
    {"encoding_policy_sha256": "0" * 64}, {"comparison_representation": "casefolded_text"},
    {"schema": audit.COMPARISON_SCHEMA}, {"decoded_digest_byte_count": True}, {"decoded_digest_byte_count": 31},
    {"digest_text_normalization_performed": True}, {"source_text_mutated": True},
    {"source_admission": True}, {"original_source_independently_verified": True}, {"PRIVATE_FIXTURE_LABEL": "forbidden"},
])
def test_v2_result_binds_exact_encoding_policy_and_retains_false_authority(tmp_path, change):
    result = audit.compare_declared_source_hash_bytes(**fixture_comparison(tmp_path))
    with pytest.raises(audit.ManifestStructureAuditError, match="^audit_metadata_refused$"):
        audit.verify_bytes_comparison_result(dict(result, **change))


def test_v2_guard_first_and_separate_explicit_cli_safe_output(tmp_path, monkeypatch, capsys):
    guard = Mock(side_effect=entry.namespace.PrivateStageError("PRIVATE_FIXTURE_BYTES_PATH"))
    bounded = Mock()
    monkeypatch.setattr(entry.namespace, "_p2_guard", guard)
    monkeypatch.setattr(entry.namespace, "_bounded_path", bounded)
    assert entry.main(["--p2-source-hash-bytes-comparison-stage"]) == 3
    failure = json.loads(capsys.readouterr().err)
    assert failure["failure_stage"] == "namespace_guard" and failure["source_admission"] is False
    bounded.assert_not_called()
    result = audit.compare_declared_source_hash_bytes(**fixture_comparison(tmp_path))
    run = Mock(return_value=dict(result, receipt_sha256="1" * 64))
    monkeypatch.setattr(entry, "run_structure_audit", run)
    assert entry.main(["--p2-source-hash-bytes-comparison-stage"]) == 0
    output = capsys.readouterr()
    public = json.loads(output.out)
    assert public["comparison_representation"] == audit.COMPARISON_BYTES_REPRESENTATION
    assert public["encoding_policy_sha256"] == audit.COMPARISON_BYTES_POLICY_SHA256
    assert public["original_source_independently_verified"] is False and public["legacy_namespace_proven"] is False
    assert "PRIVATE" not in output.out + output.err and "PUBLIC_FIXTURE_LABEL" not in output.out
    run.assert_called_once_with(compare_hash_bytes=True)


def test_v2_fresh_existing_single_p2_stage_without_private_path_work_before_guard():
    text = BYTES_SBATCH.read_text()
    assert "rayan-source-hash-bytes-comparison-${SLURM_JOB_ID}" in text
    assert "#SBATCH --mem=4G" in text and "#SBATCH --cpus-per-task=1" in text and "#SBATCH --time=00:15:00" in text
    assert text.count('exec "$UDOCKER" run') == 1 and "rayan-n0-base" in text
    assert '"$NAMESPACE_HELPER" -- /bin/bash' in text and 'exec python -I "$1" --p2-source-hash-bytes-comparison-stage' in text
    prefix = text.split('"$NAMESPACE_HELPER" -- /bin/bash')[0]
    for field in ("PRIVATE_PACKAGE_PATH", "PERSONALITY_RAW_SOURCE_PATH", "PERSONALITY_RAW_MEMBER_MAP_PATH", "PERSONALITY_REVIEWED_STRUCTURE_PATH"):
        assert f'realpath -e "${field}"' not in prefix and f'stat -c %U "${field}"' not in prefix
    for forbidden in ("pip install", "udocker setup", "compile_package", "derive_raw_inference_registry"):
        assert forbidden not in text
    bash = str(Path("C:/Program Files/Git/bin/bash.exe")) if Path("C:/Program Files/Git/bin/bash.exe").is_file() else shutil.which("bash")
    if bash:
        checked = subprocess.run([bash, "-n", str(BYTES_SBATCH)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
