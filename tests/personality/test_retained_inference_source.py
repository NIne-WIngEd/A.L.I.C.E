"""Public synthetic source construction only; never private acceptance."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import Mock
import zipfile

import pytest

from src.alice_personality.n1 import compiler as c
from src.alice_personality.n1 import retained_inference_source as retained
from tests.personality.test_identity_substrate import fixture_rows, fixture_manifest


REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/eipm/n1/inspect_retained_inference_source.py"


def package(tmp_path, *, rows=None, payload=None, corrupt_ledger=False, extra=None):
    active = fixture_rows()
    members = {name: b"".join(c._canonical(row) + b"\n" for row in active[kind])
               for kind, name in c.ACTIVE_FILES.items()}
    members.update({c.UNKNOWN_FILE: b"", c.ALTERNATIVE_FILE: b"",
                    "curation_manifest.json": c._canonical(fixture_manifest(active, 0))})
    proposals = rows if rows is not None else [
        {"proposal_id": "raw.einf.1", "provenance_class": "E-INF", "text": "PRIVATE_VALUE_SENTINEL"},
        {"proposal_id": "raw.einf.2", "provenance_class": "E-INF", "PRIVATE_KEY_SENTINEL": False}]
    members[retained.MEMBER] = payload if payload is not None else b"".join(
        c._canonical(row) + b"\n" for row in proposals)
    members.update(extra or {})
    members["SHA256SUMS.txt"] = b"".join((
        ("0" * 64 if corrupt_ledger and name == retained.MEMBER else sha256(data).hexdigest())
        + "  " + name + "\n").encode() for name, data in members.items())
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr("PUBLIC_FRONTIER/" + name, data)
    pin = c.PackagePin(sha256(path.read_bytes()).hexdigest(), "PUBLIC_FRONTIER",
        {"PUBLIC_FRONTIER/" + name: sha256(data).hexdigest() for name, data in members.items()})
    return path, pin


def test_actual_retained_ids_close_support_without_original_or_training_claim(tmp_path):
    path, pin = package(tmp_path)
    before = path.read_bytes()
    protected, summary = retained.inspect_retained_source(path, pin=pin)
    retained.verify_result(protected, summary)
    assert protected["candidate_registry"] == {"EINF": ["raw.einf.1", "raw.einf.2"]}
    assert summary["row_count"] == summary["observed_proposal_count"] == 2
    assert summary["namespace_contract"] == "exact_proposal_contract"
    counts = summary["retained_candidate_coverage"]["global_counts"]
    assert counts["total_rows"] == 9
    assert counts["reference_union_count"] == counts["reference_registry_intersection_count"] == 1
    assert counts["reference_registry_difference_count"] == 0
    assert summary["retained_candidate_coverage"]["einf_source_origin_counts"]["resolved_unique_references"] == 2
    assert protected["top_level_field_type_counts"]["PRIVATE_KEY_SENTINEL"] == {"boolean": 1}
    public = json.dumps(summary)
    assert all(s not in public for s in ("PRIVATE_KEY_SENTINEL", "PRIVATE_VALUE_SENTINEL", "raw.einf.1"))
    assert all(summary[key] is False for key in retained.FLAGS)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("rows,status", [
    ([{"id": "raw.einf.1", "provenance_class": "E-INF"}], "invalid_original_proposal_id"),
    ([{"proposal_id": "raw.einf.1", "provenance_class": "A-SYN"}], "invalid_proposal_namespace"),
    ([{"proposal_id": "raw.einf.1", "provenance_class": "E-INF"}] * 2, "duplicate_original_proposal_id")])
def test_nonconforming_rows_preserve_profile_without_alias_or_coverage(tmp_path, rows, status):
    path, pin = package(tmp_path, rows=rows)
    protected, summary = retained.inspect_retained_source(path, pin=pin)
    assert summary["namespace_contract"] == status
    assert summary["retained_candidate_coverage"] is None
    assert summary["candidate_registry_sha256"] is None
    assert protected["candidate_registry"] is None
    assert summary["observed_proposal_count"] == 0
    assert summary["row_count"] == len(rows)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{}\n\n', b'[]\n'])
def test_malformed_rows_refuse_without_source_echo(tmp_path, payload):
    path, pin = package(tmp_path, payload=payload)
    with pytest.raises(retained.RetainedSourceError) as result:
        retained.inspect_retained_source(path, pin=pin)
    assert str(result.value) == "retained_source_refused"


def test_archive_pin_refuses_before_zip_member_read(tmp_path, monkeypatch):
    path, pin = package(tmp_path)
    wrong = c.PackagePin("0" * 64, pin.package_root, pin.members_sha256)
    monkeypatch.setattr(zipfile.ZipFile, "open", Mock(side_effect=AssertionError("payload opened")))
    with pytest.raises(retained.RetainedSourceError):
        retained.inspect_retained_source(path, pin=wrong)


@pytest.mark.parametrize("kwargs", [{"corrupt_ledger": True}, {"extra": {"../unsafe": b"private"}}])
def test_ledger_and_unsafe_inventory_refuse(tmp_path, kwargs):
    path, pin = package(tmp_path, **kwargs)
    with pytest.raises(retained.RetainedSourceError):
        retained.inspect_retained_source(path, pin=pin)


def test_closing_source_and_implementation_change_refuse(tmp_path, monkeypatch):
    path, pin = package(tmp_path)
    real_coverage = c.audit_raw_lineage_coverage
    def mutate(*args, **kwargs):
        result = real_coverage(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed")
        return result
    monkeypatch.setattr(c, "audit_raw_lineage_coverage", mutate)
    with pytest.raises(retained.RetainedSourceError):
        retained.inspect_retained_source(path, pin=pin)
    path, pin = package(tmp_path)
    monkeypatch.setattr(c, "audit_raw_lineage_coverage", real_coverage)
    before = retained.implementation_hashes()
    monkeypatch.setattr(retained, "implementation_hashes", Mock(side_effect=[before, {**before, "retained.py": "0" * 64}]))
    with pytest.raises(retained.RetainedSourceError):
        retained.inspect_retained_source(path, pin=pin)


def test_public_summary_rejects_resealed_payload_and_authority(tmp_path):
    path, pin = package(tmp_path)
    protected, summary = retained.inspect_retained_source(path, pin=pin)
    for mutate in (lambda s: s.update(secret="PRIVATE_VALUE_SENTINEL"),
                   lambda s: s.update(training_authorized=True),
                   lambda s: s["retained_candidate_coverage"]["global_counts"].update(secret="PRIVATE_VALUE_SENTINEL")):
        changed = deepcopy(summary)
        mutate(changed)
        changed["receipt_sha256"] = sha256(c._canonical({k:v for k,v in changed.items() if k != "receipt_sha256"})).hexdigest()
        with pytest.raises(retained.RetainedSourceError):
            retained.verify_result(protected, changed)


def test_unicode_ids_use_actual_compiler_registry_binding(tmp_path):
    path, pin = package(tmp_path, rows=[{"proposal_id": "raw.einf.é", "provenance_class": "E-INF"}])
    protected, summary = retained.inspect_retained_source(path, pin=pin)
    assert protected["candidate_registry"] == {"EINF": ["raw.einf.é"]}
    expected = sha256('{"EINF":["raw.einf.é"]}'.encode("utf-8")).hexdigest()
    assert summary["candidate_registry_sha256"] == expected
    assert summary["retained_candidate_coverage"]["raw_lineage_registry_sha256"] == expected


def test_resealed_impossible_coverage_refuses(tmp_path):
    path, pin = package(tmp_path)
    protected, summary = retained.inspect_retained_source(path, pin=pin)
    coverage = summary["retained_candidate_coverage"]
    coverage["global_counts"]["resolved_unique_references"] = 999
    coverage["receipt_sha256"] = sha256(c._canonical({k:v for k,v in coverage.items() if k != "receipt_sha256"})).hexdigest()
    summary["receipt_sha256"] = sha256(c._canonical({k:v for k,v in summary.items() if k != "receipt_sha256"})).hexdigest()
    with pytest.raises(retained.RetainedSourceError):
        retained.verify_result(protected, summary)


def test_resealed_registry_rejects_forbidden_unicode_category(tmp_path):
    path, pin = package(tmp_path)
    protected, summary = retained.inspect_retained_source(path, pin=pin)
    protected["candidate_registry"]["EINF"][0] = "raw.einf.1\u0080"
    registry_sha = sha256(c._canonical(protected["candidate_registry"])).hexdigest()
    summary["candidate_registry_sha256"] = registry_sha
    summary["protected_sha256"] = sha256(c._canonical(protected)).hexdigest()
    coverage = summary["retained_candidate_coverage"]
    coverage["raw_lineage_registry_sha256"] = registry_sha
    coverage["receipt_sha256"] = sha256(c._canonical({k:v for k,v in coverage.items() if k != "receipt_sha256"})).hexdigest()
    summary["receipt_sha256"] = sha256(c._canonical({k:v for k,v in summary.items() if k != "receipt_sha256"})).hexdigest()
    with pytest.raises(retained.RetainedSourceError):
        retained.verify_result(protected, summary)


def load_entry():
    spec = importlib.util.spec_from_file_location("retained_entry_fixture", ENTRY)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_denied_guard_precedes_all_private_path_operations(monkeypatch):
    entry = load_entry()
    def denied():
        raise entry.namespace.PrivateStageError("PRIVATE_VALUE_SENTINEL")
    monkeypatch.setattr(entry.namespace, "_p2_guard", denied)
    monkeypatch.setattr(Path, "resolve", Mock(side_effect=AssertionError("path before guard")))
    monkeypatch.setattr(Path, "stat", Mock(side_effect=AssertionError("stat before guard")))
    with pytest.raises(entry.namespace.PrivateStageError, match="PRIVATE_VALUE_SENTINEL"):
        entry.run_inspection()


def test_entry_errors_do_not_echo_exception_or_source(monkeypatch, capsys):
    entry = load_entry()
    monkeypatch.setattr(entry, "run_inspection", Mock(side_effect=ValueError("PRIVATE_VALUE_SENTINEL")))
    assert entry.main(["--p2-retained-source-stage"]) == 3
    output = capsys.readouterr()
    assert "PRIVATE_VALUE_SENTINEL" not in output.err + output.out
    result = json.loads(output.err)
    assert result == {"state": "FAILED_UNQUALIFIED", "failure_reason": "retained_source_refused",
                      **{key: False for key in retained.FLAGS}}


def test_batch_script_parses():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    result = subprocess.run([bash, "-n", str(REPO / "scripts/eipm/n1/magnolia_inspect_retained_source.sbatch")],
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def envelope_inputs(tmp_path, *, changes=None, cr_whitespace=False):
    rows = [dict(declared_source=f"raw.einf.{index}", envelope_key=f"pool.{index}",
        origin="E-INF", historical=False, trainable=False,
        original={"legacy_detail": "PRIVATE_VALUE_SENTINEL", "conditions": ["PRIVATE_VALUE_SENTINEL"]})
        for index in (1, 2)]
    if changes:
        changes(rows)
    payload = b"".join(c._canonical(row).replace(b",", b",\r") + b"\n" for row in rows) if cr_whitespace else None
    path, pin = package(tmp_path, rows=rows, payload=payload)
    previous, prior = retained.inspect_retained_source(path, pin=pin)
    profile_bytes = c._canonical(previous) + b"\n"
    layout = {"schema": "alice-personality-retained-envelope-layout-review-v1", "state": "PROPOSED_UNQUALIFIED",
        "source_archive_sha256": pin.archive_sha256,
        "retained_member_sha256": pin.members_sha256[pin.package_root + "/" + retained.MEMBER],
        "profile_file_sha256": sha256(profile_bytes).hexdigest(),
        "profile_canonical_sha256": sha256(c._canonical(previous)).hexdigest(),
        "pointers": {"declared_source_id": ["declared_source"], "pool_id": ["envelope_key"],
                     "provenance": ["origin"], "historical_authority": ["historical"],
                     "training_authority": ["trainable"], "original_record": ["original"]},
        "review_id": "public-fixture-layout-review", "selection_basis": "Actual fixture envelope declaration, no aliasing.",
        **{key: False for key in retained.FLAGS}}
    return path, pin, previous, layout


def test_reviewed_envelope_preserves_declared_namespace_and_nested_structure(tmp_path):
    path, pin, profile, layout = envelope_inputs(tmp_path)
    protected, summary = retained.inspect_retained_source(path, pin=pin, reviewed_layout=layout, reviewed_profile=profile)
    retained.verify_result(protected, summary)
    assert protected["candidate_registry"] == {"EINF": ["raw.einf.1", "raw.einf.2"]}
    assert summary["observed_proposal_count"] == 0
    assert summary["observed_envelope_identifier_count"] == 2
    assert summary["namespace_provenance"] == "member_pinned_retained_envelope_declaration"
    assert summary["namespace_contract"] == "reviewed_retained_envelope_contract"
    assert summary["retained_candidate_coverage"]["global_counts"]["reference_registry_intersection_count"] == 1
    variants = protected["nested_schema_variants"]
    assert len(variants) == 1 and variants[0]["count"] == 2
    assert variants[0]["structure"]["fields"]["original"]["fields"]["legacy_detail"] == {"type": "string"}
    assert "PRIVATE_VALUE_SENTINEL" not in json.dumps(protected)
    assert "pool.1" not in json.dumps(protected)
    assert "declared_source" not in json.dumps(summary)
    assert all(summary[key] is False for key in retained.FLAGS)


@pytest.mark.parametrize("change", [
    lambda layout: layout.update(profile_file_sha256="0" * 64),
    lambda layout: layout.update(profile_canonical_sha256="0" * 64),
    lambda layout: layout.update(source_archive_sha256="0" * 64),
    lambda layout: layout.update(retained_member_sha256="0" * 64),
    lambda layout: layout["pointers"].update(declared_source_id=["missing"]),
    lambda layout: layout["pointers"].update(declared_source_id=["envelope_key"]),
    lambda layout: layout.update(training_authorized=True)])
def test_envelope_layout_requires_exact_reviewed_source_profile_and_roles(tmp_path, change):
    path, pin, profile, layout = envelope_inputs(tmp_path)
    change(layout)
    with pytest.raises(retained.RetainedSourceError):
        retained.inspect_retained_source(path, pin=pin, reviewed_layout=layout, reviewed_profile=profile)


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update(historical=True),
    lambda rows: rows[0].update(trainable=True),
    lambda rows: rows[0].update(origin="A-SYN"),
    lambda rows: rows[0].update(declared_source="raw.einf.1\u0080"),
    lambda rows: rows[1].update(declared_source="raw.einf.1"),
    lambda rows: rows[1].update(envelope_key="pool.1"),
    lambda rows: rows[0].update(original="PRIVATE_VALUE_SENTINEL")])
def test_envelope_records_require_declared_candidate_rights_and_unique_identifiers(tmp_path, change):
    path, pin, profile, layout = envelope_inputs(tmp_path, changes=change)
    with pytest.raises(retained.RetainedSourceError):
        retained.inspect_retained_source(path, pin=pin, reviewed_layout=layout, reviewed_profile=profile)


def test_envelope_uses_same_lf_jsonl_contract_as_prior_profile(tmp_path):
    path, pin, profile, layout = envelope_inputs(tmp_path, cr_whitespace=True)
    protected, summary = retained.inspect_retained_source(path, pin=pin, reviewed_layout=layout, reviewed_profile=profile)
    assert summary["observed_envelope_identifier_count"] == 2
    assert protected["candidate_registry"] == {"EINF": ["raw.einf.1", "raw.einf.2"]}


def test_nested_profile_refuses_before_accumulating_over_budget_variants(tmp_path, monkeypatch):
    path, pin, profile, layout = envelope_inputs(tmp_path,
        changes=lambda rows: rows[1]["original"]["conditions"].append("another fixture item"))
    real_structure = retained.structural.manifest_structure
    visits = []
    def observe(row):
        visits.append(1)
        return real_structure(row)
    monkeypatch.setattr(retained.structural, "manifest_structure", observe)
    monkeypatch.setattr(retained, "MAX_PROFILE_BYTES", 300)
    with pytest.raises(retained.RetainedSourceError):
        retained.inspect_retained_source(path, pin=pin, reviewed_layout=layout, reviewed_profile=profile)
    assert len(visits) == 1
