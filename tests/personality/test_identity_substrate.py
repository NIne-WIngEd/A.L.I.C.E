"""Public-only tiny archive tests; no real identity package is opened.

These verify source custody/authority/split mechanics, not accepted teaching
targets, learned N1 behavior or private-gradient permission.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

from src.alice_personality.n1 import compiler as c


def fixture_rows():
    e0 = []
    for index in range(1, 4):
        text = f"Public fixture evidence event {index}."
        e0.append({"unit_id": f"e0.{index}", "record_ref": f"event.{index}", "text": text,
                   "text_sha256": sha256(text.encode()).hexdigest().upper(), "provenance_class": "E0",
                   "historical_truth_allowed": True, "alice_lived_memory": False,
                   "training_authority": False, "use_lanes": ["direct_identity_supervision"],
                   "loss_mask": {"direct_identity": True}, "semantic_labels": ["public.value"]})

    def inferred(index):
        return {"curated_id": f"einf.{index}", "source_proposal_id": f"raw.einf.{index}",
                "provenance_class": "E-INF", "behavioral_proposal": "Revisable public fixture inference.",
                "identity_supporting_E0_unit_ids": [f"e0.{index}"], "historical_Elaina_truth": False,
                "training_authority": False, "owner_final_review_required": True}

    def synthetic(record_id, **values):
        return {"curated_id": record_id, "provenance_class": "A-SYN", "training_authority": False,
                "historical_Elaina_truth": False, "Alice_lived_memory": False,
                "runtime_behavioral_prior_allowed": True, "autobiographical_recall_allowed": False,
                "behavioral_proposal": "Synthetic fixture starting behavior, never a memory.", **values}

    return {"E0": e0, "EINF": [inferred(1), inferred(2)],
            "ASYN_DIRECT": [synthetic("asyn.direct", identity_supporting_E0_unit_ids=["e0.1"],
                                      supporting_curated_EINF_ids=["einf.1"])],
            "ASYN_BASE": [{**synthetic("unused", context_only_E0_unit_ids=["e0.2"]),
                           "curated_base_id": "asyn.base"}],
            "ASYN_TARGETED": [synthetic("asyn.target", supporting_raw_EINF_ids=["raw.einf.1"],
                                        related_curated_ASYN_ids=["asyn.base"])],
            "ASYN_CONTEXT": [synthetic("asyn.context", excluded_context_E0_unit_ids=["e0.3"],
                                       related_curated_ASYN_ids=["asyn.direct"])]}


def fixture_manifest(rows, unknown_count=1):
    return {"authority_flags": {"E0_mutated": False, "candidate_promotion_to_E0_performed": False,
                                "model_training_performed": False, "weights_created": False,
                                "training_authority_granted": False,
                                "owner_final_preweight_review_required": True},
            "counts": {"curated_EINF": len(rows["EINF"]), "curated_direct_ASYN_carried": len(rows["ASYN_DIRECT"]),
                       "factorized_base_policies": len(rows["ASYN_BASE"]),
                       "targeted_ASYN_added": len(rows["ASYN_TARGETED"]),
                       "context_variants_added": len(rows["ASYN_CONTEXT"]),
                       "historical_UNKNOWN_bank": unknown_count, "targeted_gap_queue_remaining": 0}}


class IdentitySubstrateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.rows = fixture_rows()
        self.registry = {"EINF": ["raw.einf.1"]}
        self.index = 0

    def archive(self, *, rows=None, manifest=None, unknown=None, alternatives=None,
                extra=None, omit=(), checksum_omit=(), modes=None, duplicates=(), overrides=None):
        rows = deepcopy(self.rows if rows is None else rows)
        unknown = [{"competitor_id": "unknown.fixture", "training_authority": False}] \
            if unknown is None else unknown
        alternatives = [{"competitor_id": "alternative.fixture", "training_authority": False,
                         "hard_negative_authorized": False, "policy": "Plausible co-valid fixture branch."}] \
            if alternatives is None else alternatives
        manifest = fixture_manifest(rows, len(unknown)) if manifest is None else manifest
        contents = {relative: b"".join(c._canonical(row) + b"\n" for row in rows[kind])
                    for kind, relative in c.ACTIVE_FILES.items()}
        contents.update({c.UNKNOWN_FILE: b"".join(c._canonical(row) + b"\n" for row in unknown),
                         c.ALTERNATIVE_FILE: b"".join(c._canonical(row) + b"\n" for row in alternatives),
                         "curation_manifest.json": c._canonical(manifest) + b"\n"})
        contents.update(extra or {})
        contents.update(overrides or {})
        contents = {name: data for name, data in contents.items() if name not in omit}
        sums = "".join(sha256(data).hexdigest() + "  " + name + "\n"
                       for name, data in sorted(contents.items()) if name not in checksum_omit).encode()
        contents["SHA256SUMS.txt"] = sums
        self.index += 1
        path = self.root / f"public-package-{self.index}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo("PUBLIC_FRONTIER/" + name)
                if modes and name in modes:
                    info.create_system = 3
                    info.external_attr = modes[name] << 16
                archive.writestr(info, data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for name in duplicates:
                    archive.writestr("PUBLIC_FRONTIER/" + name, contents[name])
        pin = c.PackagePin(sha256(path.read_bytes()).hexdigest(), "PUBLIC_FRONTIER",
                           {"PUBLIC_FRONTIER/" + name: sha256(data).hexdigest()
                            for name, data in contents.items()})
        return path, pin

    def compile(self, package=None, pin=None, **kwargs):
        if package is None:
            package, pin = self.archive()
        output = self.root / f"compiled-{self.index}"
        receipt = c.compile_package(package, output, pin=pin,
                                    raw_lineage_registry=kwargs.pop("raw_lineage_registry", self.registry), **kwargs)
        return output, receipt

    def records(self, output, name="active_identity_records.jsonl"):
        return [json.loads(line) for line in (output / name).read_text(encoding="utf-8").splitlines()]

    def reseal(self, output, update):
        path = output / "compile_receipt.json"
        receipt = json.loads(path.read_bytes())
        update(receipt)
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = sha256(c._canonical(receipt)).hexdigest()
        path.write_bytes(c._canonical(receipt) + b"\n")

    def test_complete_archive_compiles_without_claiming_authority_or_behavior(self):
        package, pin = self.archive(extra={"voice_overlay_candidate.jsonl": b'{"training_authority":false}\n'})
        before = package.read_bytes()
        output, receipt = self.compile(package, pin)
        self.assertEqual(c.verify_compiled(output, expected_source_archive_sha256=pin.archive_sha256,
                                           expected_receipt_sha256=receipt["receipt_sha256"]), receipt)
        self.assertEqual(package.read_bytes(), before)
        self.assertEqual(receipt["state"], "COMPILED_UNQUALIFIED")
        self.assertFalse(receipt["private_gradient_authorized"])
        self.assertFalse(receipt["acceptance_authority"])
        self.assertFalse(receipt["source_training_authority_granted"])
        self.assertTrue(receipt["source_owner_final_preweight_review_required"])
        self.assertIsNone(receipt["behavior_qualification"])
        self.assertFalse(receipt["voice_overlay_ingested"])
        self.assertTrue(receipt["alternatives_are_unordered_not_negatives"])
        self.assertFalse(receipt["unknown_is_behavior_void"])
        self.assertEqual(receipt["active_record_count"], 9)
        self.assertEqual(receipt["training_authority_counts"], {"false": 9})
        self.assertEqual(self.records(output, "alternative_competitors_unordered.jsonl")[0]["policy"],
                         "Plausible co-valid fixture branch.")
        self.assertEqual(len(self.records(output, "historical_unknown_bank.jsonl")), 1)
        self.assertEqual(len(list(output.iterdir())), 7)

    def test_all_support_and_derivative_closure_stays_in_one_split(self):
        output, receipt = self.compile()
        records = self.records(output)
        self.assertEqual(len({record["split"] for record in records}), 1)
        self.assertEqual(len({record["source_family_id"] for record in records}), 1)
        self.assertEqual(receipt["source_family_count"], 1)
        self.assertEqual(receipt["largest_family_record_count"], 9)
        self.assertFalse(receipt["heldout_family_coverage_complete"])
        edges = self.records(output, "support_edges.jsonl")
        self.assertTrue(any(edge["target_namespace"] == "raw-EINF" for edge in edges))
        self.assertTrue(any(edge["edge_type"] == "excluded_context" for edge in edges))

    def test_declared_active_row_eligibility_survives_global_pending_acceptance(self):
        rows = deepcopy(self.rows)
        rows["E0"][0]["training_authority"] = True
        rows["EINF"][0]["training_authority"] = True
        synthetic = rows["ASYN_DIRECT"][0]
        synthetic.pop("training_authority")
        synthetic["model_training_authority"] = True
        package, pin = self.archive(rows=rows)
        output, receipt = self.compile(package, pin)
        records = {record["record_id"]: record for record in self.records(output)}
        self.assertTrue(records["e0.1"]["training_authority"])
        self.assertTrue(records["einf.1"]["training_authority"])
        self.assertTrue(records["asyn.direct"]["training_authority"])
        self.assertIs(records["asyn.direct"]["declared_authority_flags"]["model_training_authority"], True)
        self.assertEqual(receipt["training_authority_counts"], {"false": 6, "true": 3})
        self.assertFalse(receipt["source_training_authority_granted"])
        self.assertFalse(receipt["acceptance_authority"])
        self.assertFalse(receipt["private_gradient_authorized"])
        self.assertEqual(receipt["row_authority_semantics"],
                         "declared_source_eligibility_not_package_acceptance")
        self.assertFalse(records["einf.1"]["historical_truth_allowed"])
        self.assertFalse(records["asyn.direct"]["alice_lived_memory"])
        self.assertEqual(c.verify_compiled(output), receipt)

    def test_source_order_does_not_change_connected_family_splits(self):
        output, _ = self.compile()
        first = self.records(output, "split_families.jsonl")
        reordered = {kind: list(reversed(rows)) for kind, rows in self.rows.items()}
        package, pin = self.archive(rows=reordered)
        second, _ = self.compile(package, pin)
        self.assertEqual(first, self.records(second, "split_families.jsonl"))

    def test_declared_generator_template_and_source_families_connect_records(self):
        rows = deepcopy(self.rows)
        rows["E0"].append({**rows["E0"][0], "unit_id": "isolated.e0", "record_ref": "isolated.event",
                           "generator_family_id": "shared.generator", "template_family_id": "shared.template"})
        rows["ASYN_DIRECT"][0]["generator_family_id"] = "shared.generator"
        package, pin = self.archive(rows=rows)
        output, receipt = self.compile(package, pin)
        self.assertEqual(receipt["source_family_count"], 1)
        self.assertEqual(len({row["source_family_id"] for row in self.records(output)}), 1)

    def test_raw_einf_requires_explicit_namespace_registry(self):
        for registry in (None, {}, {"EINF": ["some.other.raw"]}, {"ASYN": ["raw.einf.1"]}):
            package, pin = self.archive()
            with self.subTest(registry=registry), self.assertRaises(c.IdentitySubstrateError):
                self.compile(package, pin, raw_lineage_registry=registry)
            self.assertFalse((self.root / f"compiled-{self.index}").exists())

    def test_wrong_namespace_missing_support_and_duplicate_ids_fail(self):
        changes = [("ASYN_DIRECT", "supporting_curated_EINF_ids", ["missing.einf"]),
                   ("ASYN_DIRECT", "identity_supporting_E0_unit_ids", ["einf.1"]),
                   ("ASYN_TARGETED", "related_curated_ASYN_ids", ["e0.1"]),
                   ("EINF", "curated_id", "e0.1")]
        for kind, field, value in changes:
            rows = deepcopy(self.rows)
            rows[kind][0][field] = value
            package, pin = self.archive(rows=rows)
            with self.subTest(field=field), self.assertRaises(c.IdentitySubstrateError):
                self.compile(package, pin)

    def test_manifest_booleans_and_required_counts_are_strict(self):
        for value in ("false", 0, 1, None):
            manifest = fixture_manifest(self.rows)
            manifest["authority_flags"]["training_authority_granted"] = value
            package, pin = self.archive(manifest=manifest)
            with self.subTest(value=value), self.assertRaisesRegex(c.IdentitySubstrateError, "Booleans"):
                self.compile(package, pin)
        for mutate in (lambda m: m["counts"].pop("targeted_gap_queue_remaining"),
                       lambda m: m["counts"].update(curated_EINF=True),
                       lambda m: m["counts"].update(targeted_gap_queue_remaining=1),
                       lambda m: m["authority_flags"].pop("owner_final_preweight_review_required")):
            manifest = fixture_manifest(self.rows)
            mutate(manifest)
            package, pin = self.archive(manifest=manifest)
            with self.assertRaises(c.IdentitySubstrateError):
                self.compile(package, pin)

    def test_row_authority_history_memory_and_loss_masks_are_strict(self):
        cases = [("ASYN_DIRECT", "training_authority", "false"),
                 ("EINF", "owner_final_review_required", 1),
                 ("ASYN_DIRECT", "runtime_behavioral_prior_allowed", "true"),
                 ("ASYN_DIRECT", "historical_Elaina_truth", True),
                 ("ASYN_DIRECT", "Alice_lived_memory", True),
                 ("ASYN_DIRECT", "autobiographical_recall_allowed", True),
                 ("E0", "loss_mask", {"direct_identity": "true"}),
                 ("E0", "loss_mask", {"direct_identity": 1}),
                 ("E0", "provenance_class", "A-SYN")]
        for kind, field, value in cases:
            rows = deepcopy(self.rows)
            rows[kind][0][field] = value
            package, pin = self.archive(rows=rows)
            with self.subTest(kind=kind, field=field), self.assertRaises(c.IdentitySubstrateError):
                self.compile(package, pin)

    def test_conflicting_training_flags_and_explicit_excluded_identity_masks_fail(self):
        rows = deepcopy(self.rows)
        rows["E0"][0]["model_training_authority"] = True
        package, pin = self.archive(rows=rows)
        with self.assertRaisesRegex(c.IdentitySubstrateError, "conflict"):
            self.compile(package, pin)
        rows = deepcopy(self.rows)
        rows["E0"][0]["use_lanes"] = ["exclude_from_identity_loss"]
        package, pin = self.archive(rows=rows)
        with self.assertRaisesRegex(c.IdentitySubstrateError, "identity loss"):
            self.compile(package, pin)
        rows = deepcopy(self.rows)
        rows["ASYN_DIRECT"][0]["context_only_E0_unit_ids"] = ["e0.1"]
        package, pin = self.archive(rows=rows)
        with self.assertRaisesRegex(c.IdentitySubstrateError, "overlap"):
            self.compile(package, pin)

    def test_loss_role_diagnostic_is_aggregate_and_never_grants_authority(self):
        rows = deepcopy(self.rows)
        rows["E0"][0]["use_lanes"] = ["context_only_conditioning"]
        rows["E0"][0]["loss_mask"] = {"direct_identity": True}
        rows["E0"][1]["use_lanes"] = ["exclude_from_identity_loss"]
        rows["E0"][1]["loss_mask"] = {"direct_identity": False, "exclude_from_identity_loss": True,
                                      "PRIVATE_LOOKING_FIXTURE_KEY": True}
        package, pin = self.archive(rows=rows)
        report = c.audit_loss_role_structure(package, pin=pin)
        counts = report["counts"]
        self.assertEqual(counts["e0_rows"], 3)
        self.assertEqual(counts["legacy_conflicting_rows"], 2)
        self.assertEqual(counts["conflicting_positive_direct_identity_rows"], 1)
        self.assertEqual(counts["conflicting_identity_exclusion_name_rows"], 1)
        self.assertEqual(counts["positive_known_identity_rows"], 2)
        self.assertEqual(counts["context_and_positive_known_identity_rows"], 1)
        self.assertEqual(counts["explicit_exclusion_rows"], 1)
        self.assertEqual(counts["explicit_identity_loss_conflicting_rows"], 0)
        self.assertEqual(counts["positive_known_fields_under_exclusion_lane"],
                         {key: 0 for key in c._POSITIVE_IDENTITY_LOSS_MASKS})
        self.assertFalse(report["acceptance_authority"])
        self.assertFalse(report["training_authorized"])
        rendered = json.dumps(report)
        self.assertNotIn("PRIVATE_LOOKING_FIXTURE_KEY", rendered)
        self.assertNotIn("Public fixture evidence", rendered)
        self.assertNotIn("e0.1", rendered)
        output, receipt = self.compile(package, pin)
        records = {record["record_id"]: record for record in self.records(output)}
        for row in rows["E0"]:
            self.assertEqual(records[row["unit_id"]]["loss_mask"], row["loss_mask"])
            self.assertEqual(records[row["unit_id"]]["supervision_lanes"], row["use_lanes"])
        self.assertFalse(receipt["acceptance_authority"])
        self.assertFalse(receipt["private_gradient_authorized"])
        with self.assertRaisesRegex(c.IdentitySubstrateError, "archive SHA256"):
            c.audit_loss_role_structure(package, pin=c.PackagePin("0" * 64, pin.package_root, pin.members_sha256))

    def test_context_conditioning_and_direct_or_conditional_loss_are_independent_declared_uses(self):
        for mask in ({"direct_identity": True}, {"direct_identity": False, "conditional_identity": True},
                     {"direct_identity": True, "conditional_identity": True},
                     {"identity_core": True, "identity_loss": True}):
            rows = deepcopy(self.rows)
            rows["E0"][0]["use_lanes"] = ["context_only_conditioning", "direct_identity_supervision", "public_fixture_other_use"]
            rows["E0"][0]["loss_mask"] = mask
            package, pin = self.archive(rows=rows)
            with self.subTest(mask=mask):
                output, receipt = self.compile(package, pin)
                record = next(record for record in self.records(output) if record["record_id"] == "e0.1")
                self.assertEqual(record["loss_mask"], mask)
                self.assertEqual(record["supervision_lanes"], rows["E0"][0]["use_lanes"])
                self.assertFalse(receipt["source_training_authority_granted"])
                self.assertFalse(receipt["acceptance_authority"])
                self.assertFalse(receipt["private_gradient_authorized"])
                self.assertEqual(c.verify_compiled(output), receipt)

    def test_explicit_exclusion_lane_or_mask_rejects_each_exact_known_positive(self):
        for field in c._POSITIVE_IDENTITY_LOSS_MASKS:
            for exclusion in ("lane", "mask", "both"):
                rows = deepcopy(self.rows)
                rows["E0"][0]["use_lanes"] = ["context_only_conditioning"]
                rows["E0"][0]["loss_mask"] = {field: True}
                if exclusion in {"lane", "both"}:
                    rows["E0"][0]["use_lanes"].append("exclude_from_identity_loss")
                if exclusion in {"mask", "both"}:
                    rows["E0"][0]["loss_mask"]["exclude_from_identity_loss"] = True
                package, pin = self.archive(rows=rows)
                with self.subTest(field=field, exclusion=exclusion), self.assertRaisesRegex(c.IdentitySubstrateError, "identity loss"):
                    self.compile(package, pin)

    def test_exclusion_markers_and_unknown_identity_names_are_preserved_not_positive_losses(self):
        rows = deepcopy(self.rows)
        rows["E0"][0]["use_lanes"] = ["exclude_from_identity_loss", "context_only_conditioning"]
        rows["E0"][0]["loss_mask"] = {"direct_identity": False, "conditional_identity": False,
            "exclude_from_identity_loss": True, "exclude_identity_supervision": True,
            "identity_reconstruction": True, "UNKNOWN_IDENTITY_FIXTURE": True,
            "public_fixture_extra_mask": False}
        package, pin = self.archive(rows=rows)
        before = package.read_bytes()
        output, receipt = self.compile(package, pin)
        record = next(record for record in self.records(output) if record["record_id"] == "e0.1")
        self.assertEqual(record["loss_mask"], rows["E0"][0]["loss_mask"])
        self.assertEqual(record["supervision_lanes"], rows["E0"][0]["use_lanes"])
        self.assertNotIn("identity_core_allowed", record)
        self.assertEqual(package.read_bytes(), before)
        self.assertFalse(receipt["private_gradient_authorized"])
        self.assertFalse(receipt["acceptance_authority"])
        report = c.audit_loss_role_structure(package, pin=pin)
        self.assertEqual(report["counts"]["explicit_identity_loss_conflicting_rows"], 0)
        self.assertNotIn("UNKNOWN_IDENTITY_FIXTURE", json.dumps(report))

    def test_false_exclusion_mask_does_not_create_an_exclusion_or_infer_unknown_lanes(self):
        row = {"loss_mask": {"direct_identity": True, "exclude_from_identity_loss": False,
                             "identity_prompt_authorized": True}}
        mask = deepcopy(row["loss_mask"])
        lanes = ["public_fixture_unknown_identity_lane", "context_only_conditioning"]
        self.assertEqual(c._loss_mask(row, lanes), mask)
        self.assertEqual(row["loss_mask"], mask)
        self.assertEqual(lanes, ["public_fixture_unknown_identity_lane", "context_only_conditioning"])

    def test_precise_exclusion_diagnostics_are_fixed_known_counts_not_substring_guesses(self):
        rows = deepcopy(self.rows)
        rows["E0"][0]["use_lanes"] = ["exclude_from_identity_loss", "context_only_conditioning"]
        rows["E0"][0]["loss_mask"] = {"direct_identity": True, "conditional_identity": True,
                                      "exclude_from_identity_loss": True, "PRIVATE_LOOKING_FIXTURE_identity_loss": True}
        rows["E0"][1]["use_lanes"] = ["context_only_conditioning"]
        rows["E0"][1]["loss_mask"] = {"identity_core": True, "identity_loss": True, "exclude_from_identity_loss": True}
        rows["E0"][2]["use_lanes"] = ["exclude_from_identity_loss"]
        rows["E0"][2]["loss_mask"] = {"direct_identity": False, "conditional_identity": False,
            "exclude_from_identity_loss": True, "PRIVATE_LOOKING_FIXTURE_other_identity": True}
        package, pin = self.archive(rows=rows)
        before = package.read_bytes()
        report = c.audit_loss_role_structure(package, pin=pin)
        counts = report["counts"]
        self.assertEqual(counts["positive_known_fields_under_exclusion_lane"],
                         {"direct_identity": 1, "conditional_identity": 1, "identity_core": 0, "identity_loss": 0})
        self.assertEqual(counts["positive_known_fields_under_exclusion_mask"],
                         {key: 1 for key in c._POSITIVE_IDENTITY_LOSS_MASKS})
        self.assertEqual(counts["positive_known_fields_under_explicit_exclusion"],
                         {key: 1 for key in c._POSITIVE_IDENTITY_LOSS_MASKS})
        self.assertEqual(counts["exclusion_lane_rows"], 2)
        self.assertEqual(counts["exclusion_mask_rows"], 3)
        self.assertEqual(counts["explicit_exclusion_rows"], 3)
        self.assertEqual(counts["explicit_identity_loss_conflicting_rows"], 2)
        self.assertFalse(report["acceptance_authority"])
        self.assertFalse(report["training_authorized"])
        self.assertEqual(package.read_bytes(), before)
        self.assertNotIn("PRIVATE_LOOKING_FIXTURE", json.dumps(report))
        self.assertNotIn("e0.1", json.dumps(report))

    def test_unknown_and_alternatives_cannot_gain_learning_or_negative_authority(self):
        for field, value in (("training_authority", True), ("hard_negative_authorized", True),
                             ("hard_negative_authorized", "false"), ("is_training_negative", True),
                             ("acceptance_authority", True)):
            package, pin = self.archive(alternatives=[{"competitor_id": "alternative", field: value}])
            with self.subTest(field=field), self.assertRaises(c.IdentitySubstrateError):
                self.compile(package, pin)
        package, pin = self.archive(unknown=[{"competitor_id": "unknown", "historical_truth_allowed": True}])
        with self.assertRaisesRegex(c.IdentitySubstrateError, "UNKNOWN"):
            self.compile(package, pin)

    def test_e0_hash_and_duplicate_json_fields_fail_without_payload_exposure(self):
        rows = deepcopy(self.rows)
        rows["E0"][0]["text"] = "PRIVATE_LOOKING_FIXTURE_SENTINEL"
        package, pin = self.archive(rows=rows)
        with self.assertRaises(c.IdentitySubstrateError) as error:
            self.compile(package, pin)
        self.assertNotIn("PRIVATE_LOOKING_FIXTURE_SENTINEL", str(error.exception))
        package, pin = self.archive(overrides={c.UNKNOWN_FILE: b'{"training_authority":false,"training_authority":true}\n'})
        with self.assertRaises(c.IdentitySubstrateError):
            self.compile(package, pin)

    def test_archive_pin_and_exact_membership_are_enforced(self):
        package, pin = self.archive()
        wrong = c.PackagePin("0" * 64, pin.package_root, pin.members_sha256)
        with self.assertRaisesRegex(c.IdentitySubstrateError, "archive SHA256"):
            self.compile(package, wrong)
        altered = dict(pin.members_sha256)
        altered["PUBLIC_FRONTIER/curation_manifest.json"] = "0" * 64
        with self.assertRaisesRegex(c.IdentitySubstrateError, "member hash"):
            self.compile(package, c.PackagePin(pin.archive_sha256, pin.package_root, altered))
        with zipfile.ZipFile(package, "a") as archive:
            archive.writestr("PUBLIC_FRONTIER/unexpected.txt", b"unexpected")
        changed_pin = c.PackagePin(sha256(package.read_bytes()).hexdigest(), pin.package_root, pin.members_sha256)
        with self.assertRaisesRegex(c.IdentitySubstrateError, "membership differs"):
            self.compile(package, changed_pin)

    def test_missing_required_members_and_incomplete_checksums_fail(self):
        package, pin = self.archive(omit=(c.ACTIVE_FILES["ASYN_CONTEXT"],))
        with self.assertRaisesRegex(c.IdentitySubstrateError, "omits required"):
            self.compile(package, pin)
        package, pin = self.archive(extra={"metadata.txt": b"public"}, checksum_omit=("metadata.txt",))
        with self.assertRaisesRegex(c.IdentitySubstrateError, "complete archive membership"):
            self.compile(package, pin)

    def test_unsafe_archive_paths_links_duplicates_and_devices_are_rejected(self):
        for name in ("../escape.txt", "folder/../../escape.txt", "/absolute.txt", "C:/drive.txt",
                     "folder\\windows.txt", "folder//ambiguous.txt"):
            package, pin = self.archive(extra={name: b"public"})
            with self.subTest(name=name), self.assertRaisesRegex(c.IdentitySubstrateError, "unsafe"):
                self.compile(package, pin)
        self.assertFalse((self.root / "escape.txt").exists())
        for mode in (stat.S_IFLNK | 0o777, stat.S_IFCHR | 0o600, stat.S_IFIFO | 0o600):
            package, pin = self.archive(modes={"curation_manifest.json": mode})
            with self.subTest(mode=mode), self.assertRaisesRegex(c.IdentitySubstrateError, "links and special"):
                self.compile(package, pin)
        package, pin = self.archive(duplicates=("curation_manifest.json",))
        with self.assertRaisesRegex(c.IdentitySubstrateError, "duplicate archive"):
            self.compile(package, pin)

    def test_output_is_append_only_and_failure_cannot_replace_existing_work(self):
        output, receipt = self.compile()
        existing = {path.name: path.read_bytes() for path in output.iterdir()}
        package = Path(receipt["source_package_path"])
        pin = c.PackagePin(receipt["source_package_sha256"], receipt["package_pin"]["package_root"],
                           receipt["package_pin"]["members_sha256"])
        with self.assertRaisesRegex(c.IdentitySubstrateError, "new directory"):
            c.compile_package(package, output, pin=pin, raw_lineage_registry=self.registry)
        self.assertEqual(existing, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_reverification_detects_output_source_membership_and_receipt_tampering(self):
        for target in ("output", "source", "membership", "receipt"):
            output, receipt = self.compile()
            if target == "output":
                (output / "active_identity_records.jsonl").write_bytes(b"changed\n")
            elif target == "source":
                Path(receipt["source_package_path"]).write_bytes(b"changed source")
            elif target == "membership":
                (output / "unexpected.txt").write_text("public", encoding="utf-8")
            else:
                (output / "compile_receipt.json").write_text("{}", encoding="utf-8")
            with self.subTest(target=target), self.assertRaises(c.IdentitySubstrateError):
                c.verify_compiled(output)

    def test_resealed_authority_or_qualification_claims_still_fail(self):
        changes = [{"private_gradient_authorized": True}, {"private_gradient_authorized": 0},
                   {"acceptance_authority": True}, {"state": "APPROVED"}, {"behavior_qualification": "PASS"},
                   {"alternatives_are_unordered_not_negatives": 1}, {"voice_overlay_ingested": True},
                   {"source_training_authority_granted": True}]
        for change in changes:
            output, _ = self.compile()
            self.reseal(output, lambda receipt: receipt.update(change))
            with self.subTest(change=change), self.assertRaises(c.IdentitySubstrateError):
                c.verify_compiled(output)

    def test_caller_bound_receipt_and_source_ids_cannot_be_substituted(self):
        output, _ = self.compile()
        with self.assertRaisesRegex(c.IdentitySubstrateError, "caller-bound"):
            c.verify_compiled(output, expected_receipt_sha256="0" * 64)
        with self.assertRaisesRegex(c.IdentitySubstrateError, "source archive identity"):
            c.verify_compiled(output, expected_source_archive_sha256="0" * 64)

    def test_reviewed_v2_pin_is_metadata_only_and_covers_every_named_member(self):
        pin = c.curated_frontier_v2_pin()
        self.assertEqual(pin.archive_sha256,
                         "3867ff04d1e326086b9086b2f106b9156b3a3ec8d637d3161e7bf01616183ee9")
        self.assertEqual(len(pin.members_sha256), 36)
        self.assertIn(pin.package_root + "/targeted_round_v1_raw/SHA256SUMS.txt", pin.members_sha256)

    def test_cli_compile_and_verify_emit_metadata_without_source_payloads(self):
        package, pin = self.archive()
        pin_path = self.root / "pin.json"
        pin_path.write_bytes(c._canonical(c._pin(pin)))
        registry_path = self.root / "lineage.json"
        registry_path.write_bytes(c._canonical(self.registry))
        output = self.root / "cli-compiled"
        script = Path(__file__).resolve().parents[2] / "scripts/eipm/n1/compile_identity_substrate.py"
        for args in (["compile", str(package), str(output), "--pin-json", str(pin_path),
                      "--raw-lineage-json", str(registry_path)], ["verify", str(output)]):
            result = subprocess.run([sys.executable, str(script), *args], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(result.stdout)
            self.assertEqual(metadata["state"], "COMPILED_UNQUALIFIED")
            self.assertFalse(metadata["private_gradient_authorized"])
            self.assertFalse(metadata["acceptance_authority"])
            self.assertNotIn("Public fixture evidence", result.stdout)


if __name__ == "__main__":
    unittest.main()
