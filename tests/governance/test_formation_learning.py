from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from cognitive_kernel.formation_gold import load_frozen_formation_gold
from cognitive_kernel.formation_learning import (
    CURRICULUM_SCHEMA, MIXTURE_SCHEMA, TARGET_SCHEMA, PretrainedFormationCandidate,
    admitted_rows, curriculum_rows, learning_example_from_record, mixture_rows, output_record,
    prompt_for_text_backbone,
)
from scripts.mfm.train_formation_model import (
    evaluate_development, supervised_tokens, verify_artifact_receipt,
    write_artifact_receipt,
)


MANIFEST = (Path(__file__).resolve().parents[1] /
            "fixtures/mfm/fictional_formation_gold_v1.manifest.json")


def case_record(index: int = 0) -> dict:
    # This public case exists only in this harness; no production trainer
    # receives the frozen diagnostic fixture through this test.
    case = load_frozen_formation_gold(MANIFEST)[index]
    proposals = [dict(p.record(), disposition_scope_ref="example-scope")
                 for p in case.gold.expected]
    cited = sorted({ref for p in case.gold.expected for ref in p.evidence_refs})
    return {"schema": CURRICULUM_SCHEMA, "split": "train",
            "authorization_id": "owner-directed-mfm",
            "case_id": case.gold.case_id,
            "context": case.gold.context.metadata_record(),
            "sources": [{"ref_id": ref, "content_b64": b64encode(text.encode()).decode()}
                        for ref, text in case.texts],
            "target": {"schema": TARGET_SCHEMA,
                       "proposals": proposals,
                       "dispositions": [{"scope_ref": "example-scope", "action": "propose",
                                         "evidence_refs": cited, "target_refs": []}]}}


class ToyTokenizer:
    chat_template = "toy-native-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        serialized = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            serialized += "<assistant>"
        else:
            serialized += "<end>"
        return [ord(c) for c in serialized]


class LearningBoundaryTests(unittest.TestCase):
    def test_owner_authorized_train_only_and_exact_source_digest(self):
        row = case_record()
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "synthetic.jsonl"
            raw = (json.dumps(row) + "\n").encode()
            file.write_bytes(raw)
            cases = list(curriculum_rows(file, expected_sha256=sha256(raw).hexdigest(),
                                         owner_authorization_ref="owner-directed-mfm"))
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, row["case_id"])
            with self.assertRaisesRegex(CognitiveKernelContractError, "frozen digest"):
                list(curriculum_rows(file, expected_sha256="0" * 64,
                                     owner_authorization_ref="owner-directed-mfm"))
            row["sources"][0]["content_b64"] = b64encode(b"tampered").decode()
            with self.assertRaisesRegex(CognitiveKernelContractError, "digest mismatch"):
                learning_example_from_record(row)
            row["split"] = "final"
            with self.assertRaisesRegex(CognitiveKernelContractError, "split"):
                learning_example_from_record(row)
        reordered = case_record(1)
        reordered["sources"].reverse()
        with self.assertRaisesRegex(CognitiveKernelContractError, "evidence order"):
            learning_example_from_record(reordered)

    def test_exact_model_output_and_grounding(self):
        case = learning_example_from_record(case_record())
        expected = json.dumps(output_record(case.target))
        candidate = PretrainedFormationCandidate("a" * 64, lambda _: expected,
                                                 "learned-inference")
        actual = candidate.infer(context=case.context, opened_sources=case.opened_sources)
        self.assertEqual(actual.model_artifact_digest, "a" * 64)
        self.assertEqual(actual.proposals[0].value_text, case.target.proposals[0].value_text)
        with self.assertRaisesRegex(CognitiveKernelContractError, "exact JSON"):
            PretrainedFormationCandidate("a" * 64, lambda _: "```json\n{}\n```",
                                         "learned-inference").infer(
                context=case.context, opened_sources=case.opened_sources)
        with self.assertRaisesRegex(CognitiveKernelContractError, "digest mismatch"):
            candidate.infer(context=case.context,
                            opened_sources=((case.opened_sources[0][0], b"other"),
                                            *case.opened_sources[1:]))

    def test_text_backbone_refuses_media_and_unbounded_truncation(self):
        case = learning_example_from_record(case_record())
        tokens = supervised_tokens(ToyTokenizer(), case, 500_000)
        self.assertGreater(tokens["labels"].count(-100), 0)
        self.assertGreater(sum(x != -100 for x in tokens["labels"]), 0)
        self.assertEqual(len(tokens["labels"]), len(tokens["input_ids"]))
        with self.assertRaisesRegex(CognitiveKernelContractError, "do not truncate"):
            supervised_tokens(ToyTokenizer(), case, 5)
        ref = replace(case.context.evidence[0], modality="image")
        altered_context = replace(case.context,
                                  evidence=(ref, *case.context.evidence[1:]))
        with self.assertRaisesRegex(CognitiveKernelContractError, "cannot consume non-text"):
            prompt_for_text_backbone(altered_context, case.opened_sources)

    def test_model_artifact_manifest_detects_changes(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "model"
            model.mkdir()
            (model / "config.json").write_bytes(b"{}")
            digest = write_artifact_receipt(model, {"test": True})
            self.assertEqual(verify_artifact_receipt(model), digest)
            (model / "config.json").write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(CognitiveKernelContractError, "differs"):
                verify_artifact_receipt(model)

    def test_development_reports_disposition_and_proposal_counts(self):
        case = learning_example_from_record(case_record())
        expected = json.dumps(output_record(case.target))
        candidate = PretrainedFormationCandidate("a" * 64, lambda _: expected,
                                                 "development-test")
        report = evaluate_development(candidate, (case,))
        self.assertEqual(report["totals"]["proposal_false_negatives"], 0)
        self.assertEqual(report["totals"]["disposition_false_negatives"], 0)
        broken = PretrainedFormationCandidate("a" * 64, lambda _: "invalid",
                                              "development-test")
        invalid = evaluate_development(broken, (case,))
        self.assertEqual(invalid["totals"]["invalid_outputs"], 1)

    def test_frozen_mixture_rejects_duplicate_cases_and_generator_mismatch(self):
        original = case_record()
        original["generator_family"] = "family-one"
        second = case_record(3)
        second["generator_family"] = "family-two"
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            entries = []
            for index, row in enumerate((original, second)):
                raw = (json.dumps(row) + "\n").encode()
                name = f"input-{index}.jsonl"
                (base / name).write_bytes(raw)
                entries.append({"path": name, "sha256": sha256(raw).hexdigest(),
                                "authorization_id": "owner-directed-mfm",
                                "generator_family": row["generator_family"], "case_count": 1})
            def read():
                manifest = {"schema": MIXTURE_SCHEMA, "corpus_id": "distinct-sources",
                            "inputs": entries}
                raw = json.dumps(manifest).encode()
                (base / "mixture.json").write_bytes(raw)
                return list(mixture_rows(base / "mixture.json",
                                         expected_sha256=sha256(raw).hexdigest(),
                                         owner_authorization_ref="owner-directed-mfm"))
            self.assertEqual(len(read()), 2)
            entries[1]["generator_family"] = "wrong-family"
            with self.assertRaisesRegex(CognitiveKernelContractError, "generator differs"):
                read()
            entries[1]["generator_family"] = "family-two"
            second["case_id"] = original["case_id"]
            raw = (json.dumps(second) + "\n").encode()
            (base / "input-1.jsonl").write_bytes(raw)
            entries[1]["sha256"] = sha256(raw).hexdigest()
            with self.assertRaisesRegex(CognitiveKernelContractError, "repeats"):
                read()
            conflict = deepcopy(original)
            conflict["case_id"] = "new-id-same-input"
            conflict["generator_family"] = "family-two"
            conflict["target"]["proposals"][0]["value_text"] = "Another interpretation."
            raw = (json.dumps(conflict) + "\n").encode()
            (base / "input-1.jsonl").write_bytes(raw)
            entries[1]["sha256"] = sha256(raw).hexdigest()
            with self.assertRaisesRegex(CognitiveKernelContractError, "conflicting targets"):
                read()

    def test_same_model_input_never_receives_two_labels(self):
        original = case_record()
        conflicting = deepcopy(original)
        conflicting["case_id"] = "different-case-id"
        conflicting["target"]["proposals"][0]["value_text"] = "A contradictory target."
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "conflict.jsonl"
            def read(first, second):
                raw = (json.dumps(first) + "\n" + json.dumps(second) + "\n").encode()
                path.write_bytes(raw)
                return list(curriculum_rows(path, expected_sha256=sha256(raw).hexdigest(),
                                            owner_authorization_ref="owner-directed-mfm"))
            with self.assertRaisesRegex(CognitiveKernelContractError, "conflicting targets"):
                read(original, conflicting)
            identical = deepcopy(original)
            identical["case_id"] = "another-case-id"
            with self.assertRaisesRegex(CognitiveKernelContractError, "duplicate formation model input"):
                read(original, identical)

    def test_admitted_private_train_and_development_handoff(self):
        cases = load_frozen_formation_gold(MANIFEST)
        selected = (cases[0], cases[3], cases[6])
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            rows = []
            for compiled, split in zip(selected, ("train", "development", "final")):
                case_id = compiled.gold.case_id
                refs = []
                source_digests = []
                for ref_id, text in compiled.texts:
                    raw = text.encode()
                    path = f"{case_id}/{ref_id}.txt"
                    rights_path = f"{case_id}/{ref_id}.rights.json"
                    receipt = {"schema": "mfm-source-rights-v1",
                               "issuer_id": "test-issuer", "authority_ref": "test-consent",
                               "host_family": compiled.host_family,
                               "source_id": ref_id, "source_sha256": sha256(raw).hexdigest(),
                               "formation_training": True, "formation_evaluation": True,
                               "model_distribution": True, "revoked": False}
                    rights_raw = json.dumps(receipt).encode()
                    if split != "final":
                        (base / path).parent.mkdir(parents=True, exist_ok=True)
                        (base / path).write_bytes(raw)
                        (base / rights_path).write_bytes(rights_raw)
                    refs.append({"source_id": ref_id, "path": path,
                                 "sha256": sha256(raw).hexdigest(),
                                 "rights_path": rights_path,
                                 "rights_sha256": sha256(rights_raw).hexdigest(),
                                 "parent_source_ids": []})
                    source_digests.append(sha256(raw).hexdigest())
                proposals = [dict(p.record(), disposition_scope_ref="case-memory")
                             for p in compiled.gold.expected]
                disposition = {"scope_ref": "case-memory", "action": "propose",
                               "evidence_refs": [r["source_id"] for r in refs],
                               "target_refs": []}
                target = {"schema": TARGET_SCHEMA, "source_ids": [r["source_id"] for r in refs],
                          "context": compiled.gold.context.metadata_record(),
                          "proposals": proposals, "dispositions": [disposition]}
                target_raw = json.dumps(target).encode()
                target_path = f"{case_id}/target.json"
                if split != "final":
                    (base / target_path).write_bytes(target_raw)
                target_hash = sha256(target_raw).hexdigest()
                rows.append({"case_id": case_id, "split": split,
                             "host_family": compiled.host_family,
                             "source_family": compiled.source_family,
                             "generator_family": f"test-generator-{split}",
                             "scenario_family": f"test-scenario-{split}",
                             "duplicate_group": f"test-duplicate-{split}",
                             "parent_case_ids": [], "author_id": f"test-author-{split}",
                             "sources": refs,
                             "target": {"path": target_path, "sha256": target_hash},
                             "reviews": [{"reviewer_id": f"reviewer-{split}-{i}",
                                          "blind": True, "decision": "accept",
                                          "target_sha256": target_hash,
                                          "source_sha256s": source_digests} for i in (1, 2)]})
            manifest = {"schema": "mfm-formation-corpus-v1", "corpus_id": "test-admitted",
                        "cases": rows}
            manifest_raw = json.dumps(manifest).encode()
            (base / "manifest.json").write_bytes(manifest_raw)
            admission = admit_formation_corpus(base / "manifest.json",
                                               expected_sha256=sha256(manifest_raw).hexdigest())
            train = list(admitted_rows(admission, split="train"))
            development = list(admitted_rows(admission, split="development"))
            self.assertEqual((len(train), len(development)), (1, 1))
            self.assertFalse((base / f"{selected[2].gold.case_id}/target.json").exists())
            with self.assertRaisesRegex(CognitiveKernelContractError, "FINAL"):
                list(admitted_rows(admission, split="final"))


if __name__ == "__main__":
    unittest.main()
