"""Synthea sample data remains source-only with exact original byte anchors."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.mfm.inventory_synthea_sample_sources import (
    _dated_strings, inventory_archive, isolated_resource,
)


class SyntheaSampleInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.archive = self.root / "samples.zip"
        self.member = "fhir/Simulated_Person_fixture.json"
        self.bundle = {"resourceType": "Bundle", "type": "transaction", "entry": [
            {"resource": {"resourceType": "Patient", "id": "fake-person-a",
                          "birthDate": "1980-01-01"}},
            {"resource": {"resourceType": "Encounter", "period": {
                "start": "2026-01-01T08:00:00Z", "end": "2026-01-01T09:00:00Z"},
                "subject": {"reference": "urn:uuid:fake-person-a"},
                "note": [{"text": "café \"quoted\" visit"}]}},
            {"resource": {"resourceType": "Observation",
                          "effectiveDateTime": "2026-01-02T10:00:00Z",
                          "subject": {"reference": "urn:uuid:fake-person-a"}}},
        ]}
        self.original = json.dumps(self.bundle, ensure_ascii=False, indent=2).encode()
        with ZipFile(self.archive, "w") as z:
            z.writestr(self.member, self.original)
            z.writestr("fhir/practitioner.json", json.dumps({
                "resourceType": "Bundle", "type": "transaction",
                "entry": [{"resource": {"resourceType": "Practitioner"}}]}))
        self.result = inventory_archive(self.archive,
                                        expected_archive_sha256=sha256(self.archive.read_bytes()).hexdigest(),
                                        sample_commit="a" * 40, generator_commit="b" * 40,
                                        code_license_sha256="c" * 64,
                                        notice_sha256="d" * 64)
        self.inventory = self.root / "inventory.json"
        self.inventory.write_bytes(json.dumps(self.result, sort_keys=True).encode())
        self.pin = sha256(self.inventory.read_bytes()).hexdigest()

    def test_inventory_keeps_one_generator_family_and_no_targets_or_rights_claim(self) -> None:
        self.assertEqual(self.result["patient_bundles"], 1)
        self.assertEqual(self.result["resource_type_counts"],
                         {"Encounter": 1, "Observation": 1, "Patient": 1})
        self.assertEqual(len(self.result["excluded_nonpatient_bundles"]), 1)
        self.assertFalse(self.result["training_admitted"])
        self.assertFalse(self.result["independent_development_or_final"])
        self.assertEqual(self.result["generator_revision_used_for_archived_sample"],
                         "unverified")
        self.assertFalse(self.result["formation_targets"])
        self.assertNotIn("target", self.result["source_families"][0])

    def test_byte_slice_is_exact_as_of_and_rejects_future(self) -> None:
        family = self.result["source_families"][0]["host_family"]
        receipt, raw = isolated_resource(
            self.archive, self.inventory, expected_inventory_sha256=self.pin,
            host_family=family, entry_ordinal=1, through_date="2026-01-01")
        start = receipt["original_byte_anchor"]["start"]
        end = receipt["original_byte_anchor"]["end"]
        self.assertEqual(self.original[start:end], raw)
        self.assertEqual(sha256(raw).hexdigest(), receipt["source_sha256"])
        self.assertFalse(receipt["training_admitted"])
        self.assertNotIn("fake-person-a", family)
        with self.assertRaisesRegex(ValueError, "future-bearing"):
            isolated_resource(self.archive, self.inventory,
                              expected_inventory_sha256=self.pin,
                              host_family=family, entry_ordinal=2,
                              through_date="2026-01-01")
        with self.assertRaisesRegex(ValueError, "dated encounter"):
            isolated_resource(self.archive, self.inventory,
                              expected_inventory_sha256=self.pin,
                              host_family=family, entry_ordinal=0,
                              through_date="2026-01-01")

    def test_source_mutation_or_manifest_mutation_rejected(self) -> None:
        family = self.result["source_families"][0]["host_family"]
        with self.assertRaisesRegex(ValueError, "external digest pin"):
            inventory_archive(self.archive, expected_archive_sha256="0" * 64,
                              sample_commit="a" * 40, generator_commit="b" * 40,
                              code_license_sha256="c" * 64, notice_sha256="d" * 64)
        self.inventory.write_bytes(self.inventory.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "external digest pin"):
            isolated_resource(self.archive, self.inventory,
                              expected_inventory_sha256=self.pin,
                              host_family=family, entry_ordinal=1,
                              through_date="2026-01-01")

    def test_narrative_date_is_not_hidden_from_as_of_scan(self) -> None:
        self.assertEqual(_dated_strings({"note": [{"text":
                         "Follow-up scheduled 2026-01-09; visit on 2026-01-01."}]}),
                         ["2026-01-09", "2026-01-01"])


if __name__ == "__main__":
    unittest.main()
