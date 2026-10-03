"""SYNTHETIC DOCUMENT-VERIFIER FIXTURES ONLY, NEVER SCIENTIFIC QUALIFICATION.

Every reviewer, case, measured value and ratification below is fabricated in a
temporary test directory. Production preparation verification is mocked. A
positive test exercises consistency mechanics; it proves neither real review,
real inference, honest precommitment nor personality readiness. No report is
published outside these temporary fixtures and no real source is opened.
"""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from src.alice_personality.gemma_n0 import preparation, qualification as q


FIXTURE_NOTICE = "PUBLIC SYNTHETIC DOCUMENT-VERIFIER FIXTURE ONLY; NOT SCIENTIFIC EVIDENCE"


def seal(value):
    value = deepcopy(value)
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = sha256(q._canonical(value)).hexdigest()
    return value


def digest(label):
    return sha256((FIXTURE_NOTICE + ":" + label).encode()).hexdigest()


class PublicDocumentFixture:
    """Rebuild exact artificial pins after intentionally changing assertions."""
    def __init__(self, root, monkeypatch):
        self.root = root
        self.paths = {name: root / ("PUBLIC_SYNTHETIC_FIXTURE_" + name + ".json") for name in
                      ("prepared", "natural", "oracle", "protocol", "ratification", "outcome", "review", "report", "recipe", "training", "session", "run", "state", "outputs")}
        self.prepared = {"schema": "PUBLIC_SYNTHETIC_PREPARATION_VERIFIER_FIXTURE_NOT_REAL_GEMMA",
            "repository": q.foundation.MODEL, "revision": q.foundation.REVISION, "role": "personality",
            "state": "PREPARED_UNQUALIFIED", "behavior_qualification": None,
            "model_geometry": {"hidden_size": 3840, "hidden_state_count": 49},
            "runtime": {"backend": "transformers", "dtype": "bfloat16", "torch_version": "PUBLIC_FIXTURE", "transformers_version": "PUBLIC_FIXTURE"}}
        self.sources = {}
        for source_id, source_class, task in (("natural", "natural_human_annotations", "natural_relation_matching_only"),
                ("oracle", "independently_reviewed_public_oracle", "reviewed_full_public_role_cases")):
            records = []
            for family, cells in q.ROLE_CELLS.items():
                for cell in cells:
                    if (source_id == "natural") != (family == "meaning_argument_roles" and cell == "role_direction_reference"):
                        continue
                    label = family + ":" + cell
                    records.append({"case_id": "PUBLIC_FIXTURE:" + label, "family_id": "PUBLIC_FIXTURE_DEV:" + label,
                                    "split": "DEV", "case_sha256": digest("case:" + label), "target_sha256": digest("target:" + label)})
            records.append({"case_id": "PUBLIC_FIXTURE_TRAIN:" + source_id, "family_id": "PUBLIC_FIXTURE_TRAIN_FAMILY:" + source_id,
                            "split": "TRAIN", "case_sha256": digest("train:" + source_id), "target_sha256": digest("train-target:" + source_id)})
            self.sources[source_id] = {"schema": q.SOURCE_SCHEMA, "state": "REVIEWED_PUBLIC_TARGET_SOURCE",
                "source_id": source_id, "source_revision": FIXTURE_NOTICE, "source_class": source_class, "task_scope": task,
                "author_ids": ["PUBLIC_FIXTURE_TARGET_AUTHOR:" + source_id], "reviewer_ids": ["PUBLIC_FIXTURE_TARGET_REVIEWER:" + source_id],
                "license_reviewed": True, "targets_independently_adjudicated": True, "source_family_derivative_closure_reviewed": True,
                "fixture_only": False, "private_identity_data": False, "final_payload_opened": False, "records": records}
        self.protocol = {"schema": q.PROTOCOL_SCHEMA, "state": "PRECOMMITTED_PUBLIC_ROLE_PROTOCOL", "protocol_id": FIXTURE_NOTICE,
            "author_ids": ["PUBLIC_FIXTURE_PROTOCOL_AUTHOR"], "created_at": "2020-01-01T00:00:00Z",
            "checks": {name: True for name in q.CHECKS}, "families": {}}
        self.outcome = {"schema": q.OUTCOME_SCHEMA, "state": "PASSED_PUBLIC_ROLE_EVALUATION",
            "started_at": "2020-01-03T00:00:00Z", "completed_at": "2020-01-04T00:00:00Z",
            "observed_model_geometry": self.prepared["model_geometry"], "observed_runtime": self.prepared["runtime"],
            "upstream_gradient": False, "upstream_tensor_mutation": False, "families": {}}
        for family, cells in q.ROLE_CELLS.items():
            self.protocol["families"][family], self.outcome["families"][family] = {}, {}
            for cell in cells:
                source_id = "natural" if family == "meaning_argument_roles" and cell == "role_direction_reference" else "oracle"
                record = next(record for record in self.sources[source_id]["records"] if record["case_id"] == "PUBLIC_FIXTURE:" + family + ":" + cell)
                self.protocol["families"][family][cell] = {"source_ids": [source_id], "minimum_cases": 1,
                    "scientific_basis": FIXTURE_NOTICE,
                    "metrics": {"PUBLIC_FIXTURE_METRIC": {"comparison": ">=", "threshold": 0.25,
                    "unit": "PUBLIC_FIXTURE_UNIT", "ratified_basis": FIXTURE_NOTICE}}}
                self.outcome["families"][family][cell] = {"cases": [{"source_id": source_id,
                    **{key: record[key] for key in ("case_id", "case_sha256", "target_sha256")}}],
                    "metrics": {"PUBLIC_FIXTURE_METRIC": 0.5}, "passed": True,
                    "run_id": "PUBLIC_FIXTURE_RUN", "observation_id": "PUBLIC_FIXTURE_OBSERVATION:" + family + ":" + cell}
        configuration = {"notice": FIXTURE_NOTICE, "provider_hidden_size": 3840, "provider_state_count": 49}
        self.recipe = {"schema": q.RECIPE_SCHEMA, "state": "PRECOMMITTED_PUBLIC_ROLE_EVALUATION_RECIPE",
            "recipe_id": FIXTURE_NOTICE, "created_at": "2020-01-01T00:00:00Z", "procedure": FIXTURE_NOTICE,
            "readout_configurations": {"PUBLIC_FIXTURE_READOUT": {"configuration": configuration,
                "configuration_sha256": sha256(q._canonical(configuration)).hexdigest(), "weight_format": "safetensors"}}}
        self.training = {"schema": q.TRAINING_SCHEMA, "state": "OPTIMIZED_PUBLIC_READOUT_UNQUALIFIED",
            "run_id": "PUBLIC_FIXTURE_RUN", "observed_updates": 1,
            "cases": [{"source_id": source_id, **{key: source["records"][-1][key] for key in
                ("case_id", "case_sha256", "target_sha256")}} for source_id, source in self.sources.items()],
            "upstream_gradient": False, "upstream_tensor_mutation": False}
        self.session = {"schema": "alice-personality-feature-session-v1",
            "source_class": "prepared_frozen_gemma_features_unqualified", "status": "CLOSED_SOURCE_VERIFIED",
            "session_source_recheck_passed": True, "mechanical_fixture_only": False, "qualification": "UNQUALIFIED",
            "source_acceptance_authority": False, "persistent_feature_values_retained": False,
            "frame_binding_sha256": [digest("frame:" + family + ":" + cell) for family, cells in q.ROLE_CELLS.items() for cell in cells]}
        self.run = {"schema": q.RUN_SCHEMA, "state": "COMPLETE_PUBLIC_ROLE_EVALUATION", "run_id": "PUBLIC_FIXTURE_RUN",
            "started_at": "2020-01-03T00:00:00Z", "completed_at": "2020-01-04T00:00:00Z",
            "upstream_gradient": False, "upstream_tensor_mutation": False}
        self.observation_overrides = {}
        self.run_overrides, self.training_overrides, self.session_overrides = {}, {}, {}
        self.output_overrides = {}
        # Opaque hashing fixture, deliberately not a real model/tensor archive.
        self.state_bytes = FIXTURE_NOTICE.encode()
        self.ratification = {"schema": q.RATIFICATION_SCHEMA, "state": "OWNER_RATIFIED_BEFORE_MEASUREMENT",
            "owner_review_id": FIXTURE_NOTICE, "reviewer_ids": ["PUBLIC_FIXTURE_OWNER_REVIEWER"], "reviewed_at": "2020-01-02T00:00:00Z",
            "reviewed_thresholds": True, "reviewed_targets": True, "reviewed_baselines": True, "reviewed_split_and_contamination": True}
        self.review = {"schema": q.OWNER_REVIEW_SCHEMA, "state": "OWNER_APPROVED_PUBLIC_N0_ROLE", "owner_review_id": FIXTURE_NOTICE,
            "reviewer_ids": ["PUBLIC_FIXTURE_FINAL_REVIEWER"], "reviewed_at": "2020-01-05T00:00:00Z",
            "approved_families": list(q.ROLE_CELLS), "unresolved_material_failures": [],
            "source_acceptance_authority": False, "private_gradient_authorized": False, "identity_qualified": False}
        self.report = {"schema": q.SCHEMA, "state": "QUALIFIED_PERSONALITY_N0", "role": "personality",
            "repository": q.foundation.MODEL, "revision": q.foundation.REVISION, "prepared_receipt_path": str(self.paths["prepared"]),
            "private_gradient_authorized": False, "identity_qualified": False}
        self.fake_verify = lambda path: seal(self.prepared)
        monkeypatch.setattr(preparation, "verify_prepared", self.fake_verify)
        self.write()

    def pin(self, name, value):
        raw = q._canonical(seal(value)) + b"\n"
        self.paths[name].write_bytes(raw)
        return {"path": self.paths[name].name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    def write(self):
        prepared_pin = self.pin("prepared", self.prepared)
        prepared = seal(self.prepared)
        source_pins = {key: self.pin(key, value) for key, value in self.sources.items()}
        inventory = sha256(q._canonical({key: value["sha256"] for key, value in source_pins.items()})).hexdigest()
        bound = {"prepared_receipt_file_sha256": prepared_pin["sha256"], "prepared_receipt_sha256": prepared["receipt_sha256"],
                 "source_inventory_sha256": inventory, "code_sha256": q.current_qualification_code_sha256(),
                 "fixture_only": False, "private_identity_data": False, "final_payload_opened": False}
        self.protocol.update(bound, sources=source_pins)
        self.outcome.update(bound)
        self.recipe.update(bound)
        recipe_pin = self.pin("recipe", self.recipe)
        self.protocol["evaluation_recipe"] = recipe_pin
        protocol_pin = self.pin("protocol", self.protocol)
        self.ratification["protocol_file_sha256"] = protocol_pin["sha256"]
        ratification_pin = self.pin("ratification", self.ratification)
        self.paths["state"].write_bytes(self.state_bytes)
        state_pin = {"path": self.paths["state"].name, "bytes": len(self.state_bytes), "sha256": sha256(self.state_bytes).hexdigest()}
        self.training.update(bound, recipe_file_sha256=recipe_pin["sha256"], learned_readout_state_sha256={"PUBLIC_FIXTURE_READOUT": state_pin["sha256"]})
        self.training.update(self.training_overrides)
        training_pin = self.pin("training", self.training)
        self.session.update(prepared_receipt_file_sha256=prepared_pin["sha256"], prepared_receipt_sha256=prepared["receipt_sha256"],
            implementation_sha256={name: bound["code_sha256"]["alice_personality/identity/" + name]
                                   for name in ("codec.py", "feature_producer.py", "contracts.py", "__init__.py")})
        self.session.update(self.session_overrides)
        session_pin = self.pin("session", self.session)
        observations = {}
        recorded_outputs = {}
        for family, cells in self.outcome["families"].items():
            for cell, measured in cells.items():
                output = {"notice": FIXTURE_NOTICE, "PUBLIC_FIXTURE_PREDICTION": [0.5, 0.5]}
                recorded_outputs[measured["observation_id"]] = {"cases": measured["cases"], "output_values": output}
                observations[measured["observation_id"]] = {"family": family, "cell": cell,
                    **{key: measured[key] for key in ("cases", "metrics", "passed")}, "frame_binding_sha256": [digest("frame:" + family + ":" + cell)],
                    "output_sha256": sha256(q._canonical(output)).hexdigest()}
        for key, changes in self.observation_overrides.items():
            observations[key].update(changes)
        for key, changes in self.output_overrides.items():
            recorded_outputs[key].update(changes)
        outputs_pin = self.pin("outputs", {**bound, "schema": q.OUTPUT_SCHEMA, "state": "RECORDED_UNQUALIFIED_PUBLIC_OUTPUTS",
            "run_id": "PUBLIC_FIXTURE_RUN", "recipe_file_sha256": recipe_pin["sha256"], "observations": recorded_outputs})
        self.run.update(bound, recipe_file_sha256=recipe_pin["sha256"], protocol_file_sha256=protocol_pin["sha256"],
            ratification_file_sha256=ratification_pin["sha256"], observed_model_geometry=self.outcome["observed_model_geometry"],
            observed_runtime=self.outcome["observed_runtime"], learned_readout_states={"PUBLIC_FIXTURE_READOUT": state_pin},
            training_receipt=training_pin, feature_production_receipts=[session_pin], observations=observations, output_receipt=outputs_pin)
        self.run.update(self.run_overrides)
        run_pin = self.pin("run", self.run)
        self.outcome.update(protocol_file_sha256=protocol_pin["sha256"], ratification_file_sha256=ratification_pin["sha256"],
            evaluation_runs={"PUBLIC_FIXTURE_RUN": run_pin})
        outcome_pin = self.pin("outcome", self.outcome)
        self.review.update(protocol_file_sha256=protocol_pin["sha256"], outcome_file_sha256=outcome_pin["sha256"],
                           ratification_file_sha256=ratification_pin["sha256"])
        review_pin = self.pin("review", self.review)
        self.report.update(prepared_receipt_file_sha256=prepared_pin["sha256"], prepared_receipt_sha256=prepared["receipt_sha256"],
                           protocol=protocol_pin, outcome=outcome_pin, ratification=ratification_pin, owner_review=review_pin)
        report_pin = self.pin("report", self.report)
        self.expected = {"expected_file_sha256": report_pin["sha256"],
                         "expected_prepared_file_sha256": prepared_pin["sha256"],
                         "expected_prepared_receipt_sha256": prepared["receipt_sha256"]}

    def verify(self):
        return q.verify_personality_n0_qualification(self.paths["report"], **self.expected)


@pytest.fixture
def synthetic_documents(tmp_path, monkeypatch):
    return PublicDocumentFixture(tmp_path, monkeypatch)


def test_consistent_synthetic_documents_only_exercise_verifier_mechanics(synthetic_documents):
    fixture = synthetic_documents
    before = {path: path.read_bytes() for path in fixture.paths.values()}
    result = fixture.verify()
    assert result == seal(fixture.report)
    assert result["identity_qualified"] is False and result["private_gradient_authorized"] is False
    assert {path: path.read_bytes() for path in fixture.paths.values()} == before
    assert FIXTURE_NOTICE in fixture.protocol["protocol_id"]


@pytest.mark.parametrize("family", list(q.ROLE_CELLS))
def test_every_role_family_is_mandatory(synthetic_documents, family):
    fixture = synthetic_documents
    fixture.protocol["families"].pop(family)
    fixture.write()
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("location", ["protocol", "outcome"])
def test_each_family_requires_actual_declared_subgate_coverage(synthetic_documents, location):
    fixture = synthetic_documents
    getattr(fixture, location)["families"]["generic_voice_expression"].pop("uncertain_acoustic_cues")
    fixture.write()
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("location,field,value", [
    ("report", "state", "PREPARED_UNQUALIFIED"), ("report", "role", "mfm"),
    ("report", "repository", "another/model"), ("report", "revision", "0" * 40),
    ("report", "private_gradient_authorized", True), ("report", "identity_qualified", True),
    ("outcome", "state", "PUBLIC_SEMANTIC_EXPERIMENT_UNQUALIFIED"),
    ("outcome", "upstream_gradient", True), ("outcome", "upstream_tensor_mutation", True),
    ("ratification", "reviewed_targets", False), ("ratification", "reviewed_thresholds", None),
    ("ratification", "reviewed_baselines", "true"),
    ("review", "unresolved_material_failures", ["PUBLIC_FIXTURE_UNRESOLVED_HOLE"]),
    ("review", "source_acceptance_authority", True),
])
def test_unqualified_or_unreviewed_claims_are_rejected(synthetic_documents, location, field, value):
    fixture = synthetic_documents
    getattr(fixture, location)[field] = value
    fixture.write()
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("field,value", [("fixture_only", True), ("private_identity_data", True),
    ("final_payload_opened", True), ("license_reviewed", "true"), ("targets_independently_adjudicated", False),
    ("source_family_derivative_closure_reviewed", False), ("source_class", "public_mechanical_fixture")])
def test_fixture_private_final_or_unreviewed_sources_cannot_qualify(synthetic_documents, field, value):
    fixture = synthetic_documents
    fixture.sources["oracle"][field] = value
    fixture.write()
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("location", ["protocol", "ratification", "source", "review"])
def test_self_review_and_post_result_precommitment_are_not_accepted(synthetic_documents, location):
    fixture = synthetic_documents
    if location == "protocol":
        fixture.protocol["created_at"] = "2020-01-04T00:00:00Z"
    elif location == "ratification":
        fixture.ratification["reviewer_ids"] = fixture.protocol["author_ids"]
    elif location == "source":
        fixture.sources["oracle"]["reviewer_ids"] = fixture.sources["oracle"]["author_ids"]
    else:
        fixture.review["reviewer_ids"] = fixture.sources["oracle"]["author_ids"]
    fixture.write()
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("failure", ["threshold_missing", "threshold_bool", "threshold_below", "metric_missing",
    "passed_bool_string", "no_cases", "duplicate_case", "wrong_target", "train_case", "minimum_count_bool"])
def test_observed_facts_must_reach_precommitted_metrics_on_real_declared_cases(synthetic_documents, failure):
    fixture = synthetic_documents
    expected = fixture.protocol["families"]["evidence_uncertainty_plurality"]["support_grounding"]
    measured = fixture.outcome["families"]["evidence_uncertainty_plurality"]["support_grounding"]
    if failure == "threshold_missing":
        expected["metrics"] = {}
    elif failure == "threshold_bool":
        expected["metrics"]["PUBLIC_FIXTURE_METRIC"]["threshold"] = True
    elif failure == "threshold_below":
        measured["metrics"]["PUBLIC_FIXTURE_METRIC"] = 0.1
    elif failure == "metric_missing":
        measured["metrics"] = {}
    elif failure == "passed_bool_string":
        measured["passed"] = "true"
    elif failure == "no_cases":
        measured["cases"] = []
    elif failure == "duplicate_case":
        measured["cases"] *= 2
    elif failure == "wrong_target":
        measured["cases"][0]["target_sha256"] = "0" * 64
    elif failure == "train_case":
        measured["cases"] = [{"source_id": "oracle", **{key: fixture.sources["oracle"]["records"][-1][key]
            for key in ("case_id", "case_sha256", "target_sha256")}}]
    else:
        expected["minimum_cases"] = True
    fixture.write()
    with pytest.raises(q.QualificationError):
        fixture.verify()


def test_fewrel_only_evidence_cannot_impersonate_full_role_targets(synthetic_documents):
    fixture = synthetic_documents
    fixture.sources["oracle"]["task_scope"] = "natural_relation_matching_only"
    fixture.write()
    with pytest.raises(q.QualificationError, match="FewRel-only"):
        fixture.verify()


@pytest.mark.parametrize("cross_source", [False, True])
def test_family_and_case_closure_prevents_train_dev_leakage(synthetic_documents, cross_source):
    fixture = synthetic_documents
    dev = fixture.sources["natural" if cross_source else "oracle"]["records"][0]
    fixture.sources["oracle"]["records"][-1]["family_id"] = dev["family_id"]
    fixture.write()
    with pytest.raises(q.QualificationError, match="leak"):
        fixture.verify()


@pytest.mark.parametrize("binding", ["expected_file_sha256", "expected_prepared_file_sha256", "expected_prepared_receipt_sha256"])
def test_external_report_and_prepared_bindings_must_match(synthetic_documents, binding):
    fixture = synthetic_documents
    fixture.expected[binding] = "0" * 64
    with pytest.raises(q.QualificationError):
        fixture.verify()


def test_unknown_schema_and_missing_protocol_cannot_promote_current_diagnostics(synthetic_documents):
    fixture = synthetic_documents
    for value in ({"schema": "alice-personality-gemma-n0-public-probe-receipt-v2", "qualification": "unqualified"},
                  {"schema": q.SCHEMA, "state": "QUALIFIED_PERSONALITY_N0"}):
        pin = fixture.pin("report", value)
        with pytest.raises(q.QualificationError):
            q.verify_personality_n0_qualification(fixture.paths["report"], **{**fixture.expected, "expected_file_sha256": pin["sha256"]})


@pytest.mark.parametrize("failure", ["changed_bytes", "rebound_wrong_source", "runtime", "geometry", "code"])
def test_measured_source_runtime_geometry_code_or_receipt_changes_refuse(synthetic_documents, monkeypatch, failure):
    fixture = synthetic_documents
    if failure == "changed_bytes":
        fixture.paths["oracle"].write_bytes(fixture.paths["oracle"].read_bytes() + b" ")
    elif failure == "rebound_wrong_source":
        monkeypatch.setattr(preparation, "verify_prepared", lambda path: {**seal(fixture.prepared), "receipt_sha256": "0" * 64})
    elif failure == "runtime":
        fixture.outcome["observed_runtime"] = {"backend": "PUBLIC_FIXTURE_WRONG_RUNTIME"}
        fixture.write()
    elif failure == "geometry":
        fixture.outcome["observed_model_geometry"] = {"hidden_state_count": 1, "hidden_size": 3840}
        fixture.write()
    else:
        original = q.current_qualification_code_sha256()
        monkeypatch.setattr(q, "current_qualification_code_sha256", lambda: {**original, "PUBLIC_FIXTURE_MUTATION.py": "0" * 64})
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("path", ["../outside.json", "/absolute.json", "..\\outside.json", "C:/outside.json", "oracle//file.json"])
def test_sidecars_cannot_leave_the_bounded_report_bundle(synthetic_documents, path):
    fixture = synthetic_documents
    fixture.report["protocol"]["path"] = path
    pin = fixture.pin("report", fixture.report)
    fixture.expected["expected_file_sha256"] = pin["sha256"]
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]'])
def test_duplicate_nonfinite_and_unknown_json_are_rejected(synthetic_documents, payload):
    fixture = synthetic_documents
    path = fixture.paths["report"]
    path.write_bytes(payload)
    with pytest.raises(q.QualificationError):
        q.verify_personality_n0_qualification(path, **{**fixture.expected, "expected_file_sha256": sha256(payload).hexdigest()})


def test_metadata_bound_fails_before_unbounded_read(synthetic_documents, monkeypatch):
    fixture = synthetic_documents
    monkeypatch.setattr(q, "MAX_METADATA_BYTES", 1)
    with pytest.raises(q.QualificationError, match="size"):
        fixture.verify()


def test_comparison_upper_bound_is_actual_numeric_comparison(synthetic_documents):
    fixture = synthetic_documents
    expected = fixture.protocol["families"]["evidence_uncertainty_plurality"]["conflict_plurality_calibration"]
    expected["metrics"]["PUBLIC_FIXTURE_METRIC"].update(comparison="<=", threshold=0.75)
    fixture.write()
    fixture.verify()
    expected["metrics"]["PUBLIC_FIXTURE_METRIC"]["threshold"] = 0.1
    fixture.write()
    with pytest.raises(q.QualificationError, match="threshold"):
        fixture.verify()


def test_geometry_numeric_type_alias_cannot_impersonate_exact_preparation(synthetic_documents):
    fixture = synthetic_documents
    fixture.outcome["observed_model_geometry"] = {**fixture.prepared["model_geometry"], "hidden_state_count": 49.0}
    fixture.write()
    with pytest.raises(q.QualificationError, match="geometry"):
        fixture.verify()


def test_scientific_integer_overflow_fails_with_sanitized_error():
    with pytest.raises(q.QualificationError, match="finite actual numbers"):
        q._number(10 ** 1000)


@pytest.mark.parametrize("field", ["fixture_only", "private_identity_data", "final_payload_opened"])
@pytest.mark.parametrize("location", ["protocol", "outcome"])
def test_bound_artifact_scope_flags_require_actual_false(synthetic_documents, field, location):
    fixture = synthetic_documents
    # Re-seal directly: fixture.write deliberately reinstates safe defaults.
    artifact = getattr(fixture, location)
    artifact[field] = 0
    pin = fixture.pin(location, artifact)
    fixture.report[location] = pin
    report_pin = fixture.pin("report", fixture.report)
    fixture.expected["expected_file_sha256"] = report_pin["sha256"]
    with pytest.raises(q.QualificationError):
        fixture.verify()


def test_missing_natural_annotation_cannot_be_covered_by_teacher_claims(synthetic_documents):
    fixture = synthetic_documents
    fixture.sources["natural"]["source_class"] = "independently_adjudicated_service_teacher"
    fixture.write()
    with pytest.raises(q.QualificationError, match="natural public annotation"):
        fixture.verify()


def test_one_artifact_cannot_impersonate_two_independent_review_roles(synthetic_documents):
    fixture = synthetic_documents
    fixture.report["ratification"] = fixture.report["owner_review"]
    pin = fixture.pin("report", fixture.report)
    fixture.expected["expected_file_sha256"] = pin["sha256"]
    with pytest.raises(q.QualificationError, match="independent protocol roles"):
        fixture.verify()


@pytest.mark.parametrize("failure", ["missing_states", "state_digest", "recipe_digest", "unfinished_run", "outside_interval",
    "training_state", "no_updates", "dev_training", "session_unclosed", "session_fixture", "session_code", "unknown_frame", "wrong_metrics", "missing_output_digest"])
def test_exact_readout_recipe_training_run_and_closed_feature_evidence_are_required(synthetic_documents, failure):
    fixture = synthetic_documents
    if failure == "missing_states":
        fixture.run_overrides["learned_readout_states"] = {}
    elif failure == "state_digest":
        fixture.training_overrides["learned_readout_state_sha256"] = {"PUBLIC_FIXTURE_READOUT": "0" * 64}
    elif failure == "recipe_digest":
        fixture.run_overrides["recipe_file_sha256"] = "0" * 64
    elif failure == "unfinished_run":
        fixture.run["state"] = "STARTED"
    elif failure == "outside_interval":
        fixture.run["started_at"] = "2020-01-02T00:00:00Z"
    elif failure == "training_state":
        fixture.training["state"] = "IMPLEMENTED_UNTRAINED_UNQUALIFIED"
    elif failure == "no_updates":
        fixture.training["observed_updates"] = 0
    elif failure == "dev_training":
        record = fixture.sources["oracle"]["records"][0]
        fixture.training["cases"] = [{"source_id": "oracle", **{key: record[key] for key in ("case_id", "case_sha256", "target_sha256")}}]
    elif failure == "session_unclosed":
        fixture.session["status"] = "STARTED"
    elif failure == "session_fixture":
        fixture.session["mechanical_fixture_only"] = True
    elif failure == "session_code":
        fixture.session_overrides["implementation_sha256"] = {name: "0" * 64 for name in
            ("codec.py", "feature_producer.py", "contracts.py", "__init__.py")}
    else:
        observation = "PUBLIC_FIXTURE_OBSERVATION:generic_voice_expression:prosody_meaning"
        change = {"frame_binding_sha256": ["0" * 64]} if failure == "unknown_frame" else {
            "metrics": {"PUBLIC_FIXTURE_METRIC": 0.75}} if failure == "wrong_metrics" else {"output_sha256": None}
        fixture.observation_overrides[observation] = change
    fixture.write()
    with pytest.raises(q.QualificationError):
        fixture.verify()


@pytest.mark.parametrize("change", [b"", b"PUBLIC FIXTURE CHANGED", None])
def test_exact_evaluated_state_bytes_are_rehashed_not_only_asserted(synthetic_documents, change):
    fixture = synthetic_documents
    path = fixture.paths["state"]
    path.write_bytes(change if change is not None else b"!" * len(fixture.state_bytes))
    with pytest.raises(q.QualificationError, match="readout state"):
        fixture.verify()


def test_readout_configuration_digest_and_recipe_precommitment_are_required(synthetic_documents):
    fixture = synthetic_documents
    fixture.recipe["readout_configurations"]["PUBLIC_FIXTURE_READOUT"]["configuration_sha256"] = "0" * 64
    fixture.write()
    with pytest.raises(q.QualificationError, match="configuration digest"):
        fixture.verify()
    fixture.recipe["readout_configurations"]["PUBLIC_FIXTURE_READOUT"]["configuration_sha256"] = sha256(q._canonical(
        fixture.recipe["readout_configurations"]["PUBLIC_FIXTURE_READOUT"]["configuration"])).hexdigest()
    fixture.recipe["created_at"] = "2020-01-04T00:00:00Z"
    fixture.write()
    with pytest.raises(q.QualificationError, match="precommitted"):
        fixture.verify()


def test_resealed_prediction_values_cannot_differ_from_measured_output_binding(synthetic_documents):
    fixture = synthetic_documents
    fixture.output_overrides["PUBLIC_FIXTURE_OBSERVATION:generic_voice_expression:prosody_meaning"] = {
        "output_values": {"notice": FIXTURE_NOTICE, "PUBLIC_FIXTURE_PREDICTION": [0.9, 0.1]}}
    fixture.write()
    with pytest.raises(q.QualificationError, match="observed case/output bytes"):
        fixture.verify()
