"""Admission tests with tiny synthetic public fixtures, never real FewRel rows.

External artifact pins and expected corpus counts are patched only inside these
tests. They exercise custody/schema/split rejection, not actual corpus
eligibility, natural-label correctness, frozen Gemma or personality behavior.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.eipm.gemma_n0 import admit_public_fewrel as cli
from src.alice_personality.gemma_n0 import public_fewrel as source


class PublicFewRelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "public-fixture"
        self.data.mkdir()
        self.paths = {kind: self.data / pin["name"] for kind, pin in source.ARTIFACT_PINS.items()}
        self.output = self.root / "eligibility.json"
        counts = {"train_rows": 12, "dev_rows": 6, "train_relation_count": 4,
                  "dev_relation_count": 2, "official_training_relation_count": 6}
        count_patch = patch.dict(source.EXPECTED_COUNTS, counts, clear=True)
        count_patch.start()
        self.addCleanup(count_patch.stop)
        self.pin_patch = patch.dict(source.ARTIFACT_PINS, deepcopy(source.ARTIFACT_PINS), clear=True)
        self.pin_patch.start()
        self.addCleanup(self.pin_patch.stop)
        relations = {}
        for number in range(1, 7):
            name, description = f"Fixture relation {number}", f"Public fixture meaning {number}"
            relations[f"P{number}"] = {"name": name, "description": description,
                                        "semantic_text": f"Relation name: {name}. Relation meaning: {description}"}
        self.bank = {"schema": source.BANK_SCHEMA, "source_id": source.SOURCE_ID,
                     "source_revision": source.SOURCE_REVISION,
                     "role": "TRAIN_AND_MODEL_SELECTION_ONLY_NO_FINAL_RELATIONS",
                     "relations": relations, "relation_keys_are_metadata_only": True,
                     "semantic_text_field": "semantic_text", "private_identity_data": False}
        ordered = sorted(relations, key=lambda key: sha256(f"{source.SPLIT_RULE}:{key}".encode()).hexdigest())
        self.dev_families, self.train_families = sorted(ordered[:2]), sorted(ordered[2:])
        self.rows = []
        for split, families in (("train", self.train_families), ("dev", self.dev_families)):
            for family in families:
                for sample in range(3):
                    points = [1, 4, 1] if split == "train" else [2, 3, 6]
                    pool = self.train_families if split == "train" else sorted(relations)
                    candidates = [family] + [key for key in pool if key != family][:points[sample] - 1]
                    sentence = f"Ada {split} family-{family} sample-{sample} Bea"
                    tokens = sentence.split()
                    self.rows.append({
                        "schema": source.ROW_SCHEMA,
                        "id": f"fewrel:{split}:" + sha256(sentence.encode()).hexdigest()[:20],
                        "split": split, "source_id": source.SOURCE_ID,
                        "source_revision": source.SOURCE_REVISION, "natural_source_text": True,
                        "generated_instruction_only": True, "generated_relation_label": False,
                        "sentence": sentence, "tokens": tokens,
                        "head": {"text": "Ada", "type": "Qfixture-1", "token_indices": [0]},
                        "tail": {"text": "Bea", "type": "Qfixture-2", "token_indices": [len(tokens) - 1]},
                        "instruction": source.INSTRUCTION, "candidate_relation_keys": candidates,
                        "target_relation_key": family, "target_candidate_index": 0,
                        "runtime_relation_count": len(candidates), "training_authorized": split == "train",
                        "model_selection_authorized": split == "dev", "final_validation_only": False,
                        "relation_keys_are_metadata_only": True, "private_identity_data": False})
        self.manifest = {"schema": source.MANIFEST_SCHEMA,
                         "status": "MATERIALIZED_NATURAL_RELATION_CURRICULUM_WITH_OFFICIAL_VALIDATION_FINAL_FAMILIES",
                         "source_id": source.SOURCE_ID, "source_revision": source.SOURCE_REVISION,
                         "source_git_blob_sha1": deepcopy(source.SOURCE_BLOBS), "license": "MIT",
                         "dev_family_split_rule": source.SPLIT_RULE, **counts,
                         "natural_source_text": True, "human_relation_labels": True,
                         "final_relation_descriptions_absent_from_train_dev_bank": True,
                         "private_identity_data": False, "final_training_authorized": False,
                         "final_model_selection_authorized": False,
                         "relation_ids_omitted_from_manifest": True, "final_rows_separate_artifact": True,
                         "all_official_validation_relation_families_are_final": True,
                         "final_family_subset_selected_after_observation": False,
                         "operating_cap_is_capability_ceiling": False, "max_per_relation_operating_cap": 0,
                         "train_candidate_count_points": [1, 4], "dev_candidate_count_points": [2, 3, 6]}
        self.audit = {"schema": source.AUDIT_SCHEMA, "status": "PASS_FEWREL_NATURAL_RELATION_AUDIT_V3",
                      "errors": [], **{key: value for key, value in counts.items()
                                       if key != "official_training_relation_count"},
                      "natural_source_text": True, "human_relation_labels": True,
                      "final_relation_descriptions_absent_from_train_dev_bank": True,
                      "private_identity_data": False, "final_training_authorized": False,
                      "final_model_selection_authorized": False, "training_authorized_by_audit": False,
                      "relation_overlap": {"train_dev": []}, "source_overlap": {"train_dev": 0},
                      "train_candidate_count_points": [1, 4], "dev_candidate_count_points": [2, 3, 6],
                      "dev_unseen_candidate_count_points": [2, 3, 6]}
        self._seal()

    def _seal(self):
        self.paths["rows"].write_bytes(b"".join(source._canonical(row) + b"\n" for row in self.rows))
        self.paths["bank"].write_bytes(source._canonical(self.bank) + b"\n")
        for kind in ("rows", "bank"):
            payload = self.paths[kind].read_bytes()
            pin = source.ARTIFACT_PINS[kind]
            pin.update(size=len(payload), sha256=sha256(payload).hexdigest())
            for metadata in (self.manifest, self.audit):
                metadata[f"train_dev_{kind}_sha256"] = pin["sha256"]
        for kind, metadata in (("manifest", self.manifest), ("audit", self.audit)):
            payload = source._canonical(metadata) + b"\n"
            self.paths[kind].write_bytes(payload)
            source.ARTIFACT_PINS[kind].update(size=len(payload), sha256=sha256(payload).hexdigest())

    def _admit(self, **kwargs):
        return source.admit_fewrel_train_dev(
            self.paths["rows"], self.paths["bank"], self.paths["manifest"], self.paths["audit"],
            kwargs.pop("output", self.output), declared_public=kwargs.pop("declared_public", True), **kwargs)

    def _reject(self, mutation, text):
        mutation()
        self._seal()
        with self.assertRaisesRegex(source.PublicSourceError, text):
            self._admit()
        self.assertFalse(self.output.exists())

    def test_roundtrip_checks_complete_fixture_and_stays_unqualified(self):
        before = {kind: path.read_bytes() for kind, path in self.paths.items()}
        receipt = self._admit()
        self.assertEqual(receipt, source.verify_fewrel_admission(self.output))
        self.assertEqual(receipt["state"], source.STATE)
        self.assertEqual(receipt["statistics"]["rows"], {"train": 12, "dev": 6})
        self.assertEqual(receipt["statistics"]["relation_families"], {"train": 4, "dev": 2})
        self.assertTrue(receipt["statistics"]["all_train_dev_records_checked"])
        self.assertFalse(receipt["statistics"]["final_payload_opened"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["historical_pass_authorizes_new_training"])
        self.assertFalse(receipt["n0_approved"])
        self.assertFalse(receipt["target_keys_indices_and_ids_are_model_input"])
        self.assertFalse(receipt["source_counts_define_capability_ceiling"])
        self.assertIsNone(receipt["behavior_qualification"])
        self.assertEqual(before, {kind: path.read_bytes() for kind, path in self.paths.items()})

    def test_only_four_named_files_no_sibling_enumeration_or_final_open(self):
        for name in ("final_rows.jsonl", "final_bank.json", "pid2name.json", "val_wiki.json"):
            (self.data / name).write_text("must stay unopened", encoding="utf-8")
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name in {"final_rows.jsonl", "final_bank.json", "pid2name.json", "val_wiki.json"}:
                raise AssertionError("forbidden payload opened")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch.object(Path, "rglob", side_effect=AssertionError("scan")), \
                patch.object(Path, "iterdir", side_effect=AssertionError("scan")):
            self._admit()

    def test_explicit_public_declaration_required(self):
        for declaration in (False, None, 1, "true"):
            with self.subTest(declaration=declaration), self.assertRaisesRegex(source.PublicSourceError, "declaration"):
                self._admit(declared_public=declaration)

    def test_all_external_pins_checked_before_any_row_parsing(self):
        for kind in self.paths:
            with self.subTest(kind=kind):
                original = self.paths[kind].read_bytes()
                self.paths[kind].write_bytes(original[:-1] + b"x")
                with patch.object(source, "_rows", side_effect=AssertionError("parsed before pins")), \
                        self.assertRaisesRegex(source.PublicSourceError, "SHA256"):
                    self._admit()
                self.paths[kind].write_bytes(original)

    def test_size_pin_checked_before_row_parser(self):
        self.paths["rows"].write_bytes(self.paths["rows"].read_bytes() + b"x")
        with patch.object(source, "_rows", side_effect=AssertionError("parser")), \
                self.assertRaisesRegex(source.PublicSourceError, "size"):
            self._admit()

    def test_reject_wrong_source_and_nonpublic_flags(self):
        original = deepcopy(self.rows[0])
        for field, value in (("source_revision", "wrong"), ("private_identity_data", True),
                             ("generated_relation_label", True), ("natural_source_text", 1),
                             ("relation_keys_are_metadata_only", False), ("generated_instruction_only", False)):
            with self.subTest(field=field):
                self.rows[0] = {**original, field: value}
                self._seal()
                with self.assertRaises(source.PublicSourceError):
                    self._admit()
        self.rows[0] = original

    def test_dev_never_gradient_and_non_train_dev_split_rejected(self):
        dev = next(row for row in self.rows if row["split"] == "dev")
        self._reject(lambda: dev.update(training_authorized=True), "split authority")
        dev["training_authorized"] = False
        self._reject(lambda: dev.update(split="final"), "non-TRAIN/DEV")

    def test_target_key_cannot_change_family_partition(self):
        row = self.rows[0]
        self._reject(lambda: row.update(target_relation_key=self.dev_families[0]), "deterministic")

    def test_train_candidate_pool_cannot_include_dev_family(self):
        row = next(row for row in self.rows if row["split"] == "train" and len(row["candidate_relation_keys"]) > 1)
        self._reject(lambda: row["candidate_relation_keys"].__setitem__(1, self.dev_families[0]), "candidate pool")

    def test_candidate_label_and_index_not_independent_inputs(self):
        row = self.rows[0]
        self._reject(lambda: row.update(target_candidate_index=True), "candidate pool")

    def test_duplicate_id_and_unknown_row_field_rejected(self):
        self._reject(lambda: self.rows[1].update(id=self.rows[0]["id"]), "duplicate")
        self.rows[1]["id"] = "fewrel:train:" + "a" * 20
        self._reject(lambda: self.rows[1].update(personality_authority="fixture"), "schema fields")

    def test_final_descriptions_not_permitted_in_bank(self):
        self._reject(lambda: self.bank.update(role="SEALED_FINAL_VALIDATION_RELATION_BANK"), "bank is not")

    def test_bank_opaque_id_and_semantic_drift_rejected(self):
        relation = next(iter(self.bank["relations"].values()))
        self._reject(lambda: relation.update(semantic_text="Relation meaning P123"), "opaque relation ID")

    def test_entity_positions_and_sentence_tokens_checked(self):
        row = self.rows[0]
        self._reject(lambda: row["head"].update(token_indices=[999]), "source-aligned")
        row["head"]["token_indices"] = [0]
        self._reject(lambda: row.update(sentence="not reconstructed"), "reconstruction")

    def test_complete_family_counts_are_checked_not_sampled(self):
        self._reject(lambda: self.rows.pop(), "complete row count")

    def test_exact_and_normalized_source_overlap_rejected(self):
        train = next(row for row in self.rows if row["split"] == "train")
        dev = next(row for row in self.rows if row["split"] == "dev")
        dev.update(sentence=train["sentence"], tokens=list(train["tokens"]),
                   head=deepcopy(train["head"]), tail=deepcopy(train["tail"]))
        self._seal()
        with self.assertRaisesRegex(source.PublicSourceError, "exact TRAIN/DEV source overlap"):
            self._admit()
        dev["tokens"] = [token.upper() for token in train["tokens"]]
        dev["sentence"] = " ".join(dev["tokens"])
        dev["head"]["text"] = train["head"]["text"].upper()
        dev["tail"]["text"] = train["tail"]["text"].upper()
        self._seal()
        with self.assertRaisesRegex(source.PublicSourceError, "normalized TRAIN/DEV source overlap"):
            self._admit()

    def test_normalized_sentence_overlap_reported_without_false_lexical_holdout_claim(self):
        train = next(row for row in self.rows if row["split"] == "train")
        dev = next(row for row in self.rows if row["split"] == "dev")
        dev["tokens"] = list(train["tokens"])
        dev["sentence"] = train["sentence"]
        dev["head"] = {"text": train["tokens"][1], "type": "Qfixture-3", "token_indices": [1]}
        self._seal()
        self.assertEqual(self._admit()["statistics"]["normalized_sentence_only_overlap"], 1)

    def test_receipt_append_only_and_outside_source(self):
        with self.assertRaisesRegex(source.PublicSourceError, "outside source"):
            self._admit(output=self.data / "receipt.json")
        self._admit()
        original = self.output.read_bytes()
        with self.assertRaisesRegex(source.PublicSourceError, "fresh"):
            self._admit()
        self.assertEqual(self.output.read_bytes(), original)

    def test_moved_alias_or_non_allowlisted_artifact_rejected(self):
        wrong = self.data / "final_rows.jsonl"
        wrong.write_bytes(self.paths["rows"].read_bytes())
        with self.assertRaisesRegex(source.PublicSourceError, "allowlisted"):
            source.admit_fewrel_train_dev(wrong, self.paths["bank"], self.paths["manifest"],
                                         self.paths["audit"], self.output, declared_public=True)
        separate = self.root / "separate"
        separate.mkdir()
        self.paths["bank"] = separate / "train_dev_bank.json"
        self.paths["bank"].write_bytes(source._canonical(self.bank) + b"\n")
        with self.assertRaisesRegex(source.PublicSourceError, "share one isolated"):
            self._admit()

    def test_after_receipt_rows_mutation_and_resealed_permission_forgery_rejected(self):
        receipt = self._admit()
        self.paths["rows"].write_bytes(self.paths["rows"].read_bytes()[:-1] + b"x")
        with self.assertRaisesRegex(source.PublicSourceError, "SHA256"):
            source.verify_fewrel_admission(self.output)
        self._seal()
        forged = deepcopy(receipt)
        forged.pop("receipt_sha256")
        forged["training_authorized"] = True
        forged["receipt_sha256"] = sha256(source._canonical(forged)).hexdigest()
        self.output.write_bytes(source._canonical(forged) + b"\n")
        with self.assertRaisesRegex(source.PublicSourceError, "no longer matches"):
            source.verify_fewrel_admission(self.output)

    def test_code_mutation_during_admission_rejected(self):
        paths = {}
        for name in source.IMPLEMENTATION_PATHS:
            path = self.root / name
            path.write_bytes(b"# public code fixture\n")
            paths[name] = path
        original_rows = source._rows
        def alter_code(*args):
            result = original_rows(*args)
            paths["public_fewrel.py"].write_bytes(b"# changed code fixture\n")
            return result
        with patch.dict(source.IMPLEMENTATION_PATHS, paths, clear=True), patch.object(source, "_rows", alter_code), \
                self.assertRaisesRegex(source.PublicSourceError, "implementation changed"):
            self._admit()

    def test_metadata_mutation_after_custody_before_parsing_rejected(self):
        original_bound = source._bound_files
        def alter_metadata(paths):
            files, records = original_bound(paths)
            payload = self.paths["manifest"].read_bytes().replace(b'"license":"MIT"', b'"license":"MIT" ')
            self.paths["manifest"].write_bytes(payload)
            return files, records
        with patch.object(source, "_bound_files", alter_metadata), \
                self.assertRaisesRegex(source.PublicSourceError, "metadata differs"):
            self._admit()

    def test_row_mutation_after_custody_before_parse_rejected(self):
        original_bound = source._bound_files
        def alter_rows(paths):
            files, records = original_bound(paths)
            payload = self.paths["rows"].read_bytes().replace(b'"sentence":', b'"sentence" :', 1)
            self.paths["rows"].write_bytes(payload)
            return files, records
        with patch.object(source, "_bound_files", alter_rows), \
                self.assertRaisesRegex(source.PublicSourceError, "row file changed"):
            self._admit()

    def test_canonical_metadata_does_not_accept_boolean_cardinality(self):
        self._reject(lambda: self.manifest.update(train_candidate_count_points=[True, 4]), "cardinality points")

    def test_verified_receipt_cannot_be_copied_inside_source(self):
        self._admit()
        alias = self.data / "eligibility.json"
        alias.write_bytes(self.output.read_bytes())
        with self.assertRaisesRegex(source.PublicSourceError, "outside source"):
            source.verify_fewrel_admission(alias)

    def test_returned_provenance_cannot_mutate_module_authority(self):
        self._admit()["provenance"]["license"] = "fixture mutation"
        self.assertEqual(source.PROVENANCE["license"], "MIT")

    def test_duplicate_json_fields_rejected_even_with_fixture_pin(self):
        raw = self.paths["rows"].read_bytes().replace(b'"split":"train"', b'"split":"train","split":"train"', 1)
        self.paths["rows"].write_bytes(raw)
        source.ARTIFACT_PINS["rows"].update(size=len(raw), sha256=sha256(raw).hexdigest())
        for metadata in (self.manifest, self.audit):
            metadata["train_dev_rows_sha256"] = sha256(raw).hexdigest()
        for kind, metadata in (("manifest", self.manifest), ("audit", self.audit)):
            payload = source._canonical(metadata) + b"\n"
            self.paths[kind].write_bytes(payload)
            source.ARTIFACT_PINS[kind].update(size=len(payload), sha256=sha256(payload).hexdigest())
        with self.assertRaisesRegex(source.PublicSourceError, "duplicate JSON"):
            self._admit()

    def test_cli_explicit_files_and_verify(self):
        arguments = ["admit", "--rows", str(self.paths["rows"]), "--bank", str(self.paths["bank"]),
                     "--manifest", str(self.paths["manifest"]), "--audit", str(self.paths["audit"]),
                     "--output", str(self.output), "--declared-public-train-dev"]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(arguments), 0)
            self.assertEqual(cli.main(["verify", str(self.output)]), 0)
        for line in output.getvalue().splitlines():
            self.assertFalse(json.loads(line)["training_authorized"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(arguments[:-1])


if __name__ == "__main__":
    unittest.main()
