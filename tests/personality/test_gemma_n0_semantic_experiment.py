"""CPU public fixtures exercise protocol/state mechanics, never Gemma competence.

Source/preparation verifiers and runtime are explicitly injected tiny fixtures.
Real frozen Gemma, FewRel payloads and remote execution are absent. The own
readout really optimizes Torch parameters and resumes actual optimizer/RNG.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.eipm.gemma_n0 import run_public_semantic_experiment as cli
from src.alice_personality.gemma_n0 import public_semantic_experiment as experiment
from src.alice_personality.gemma_n0.semantic_readout import SemanticReadout


class _Tokenizer:
    def __init__(self):
        self.truncate_offsets_only = False
    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("chat template forbidden")
    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False and kwargs["padding"] is False
        pieces = list(re.finditer(r"\S+", text))
        ids = [1, *[2 + int(sha256(piece.group().encode()).hexdigest()[:4], 16) for piece in pieces], 2]
        offsets = [[0, 0], *[[piece.start(), piece.end()] for piece in pieces], [0, 0]]
        if kwargs.get("return_offsets_mapping") and self.truncate_offsets_only:
            ids, offsets = ids[:-1], offsets[:-1]
        output = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = torch.tensor([offsets])
        return output


class _Provider:
    def __init__(self):
        self.calls = 0
    def extract_features(self, *, input_ids, attention_mask):
        self.calls += 1
        with torch.no_grad():
            states = tuple(torch.sin(input_ids.float()[..., None] / (3 + torch.arange(8).float())
                                     + layer).bfloat16() for layer in range(3))
        return SimpleNamespace(all_hidden_states=states, hidden_states=states[-1], attention_mask=attention_mask.bool())


class SemanticExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "public-source-fixture"
        self.data.mkdir()
        self.snapshot = self.root / "publisher-fixture"
        self.snapshot.mkdir()
        self.rows = []
        for split, families in (("train", ["P1", "P2"]), ("dev", ["P3"])):
            for family in families:
                for index in range(3):
                    sentence = "Ada shared Bea" if index == 0 else f"Ada public {split} {family.replace('P', 'relation')} case-{index} Bea"
                    head = "shared" if split == "dev" and index == 0 else "Ada"
                    candidates = ["P1", "P2"] if split == "train" else ["P2", "P3", "P1"]
                    rid = f"fewrel:{split}:" + sha256(f"{family}:{index}".encode()).hexdigest()[:20]
                    self.rows.append({"id": rid, "split": split, "target_relation_key": family,
                                      "candidate_relation_keys": candidates,
                                      "target_candidate_index": candidates.index(family),
                                      "sentence": sentence, "head": {"text": head, "type": "Qhead-fixture"},
                                      "tail": {"text": "Bea", "type": "Qtail-fixture"}})
        self.row_path = self.data / "train_dev_rows.jsonl"
        self.row_path.write_bytes(b"".join(experiment._canonical(row) + b"\n" for row in self.rows))
        self.bank_path = self.data / "train_dev_bank.json"
        self.bank_path.write_bytes(experiment._canonical({"relations": {
            "P1": {"semantic_text": "One entity is connected to another."},
            "P2": {"semantic_text": "One entity is distinct from another."},
            "P3": {"semantic_text": "One entity supports another."}}}) + b"\n")
        self.source_path, self.prepared_path = self.root / "source.json", self.root / "prepared.json"
        self.source_record = {"receipt_sha256": "a" * 64,
                              "statistics": {"normalized_sentence_only_overlap": 1},
                              "files": [{"kind": "rows", "path": str(self.row_path), "sha256": experiment._digest(self.row_path)},
                                        {"kind": "bank", "path": str(self.bank_path), "sha256": experiment._digest(self.bank_path)}]}
        self.prepared_record = {"receipt_sha256": "b" * 64, "snapshot_path": str(self.snapshot),
                                "runtime": {"backend": "transformers", "dtype": "bfloat16",
                                            "torch_version": str(torch.__version__), "transformers_version": "fixture-only"},
                                "model_geometry": {"hidden_state_count": 3, "hidden_size": 8,
                                                   "num_hidden_layers": 2, "max_position_embeddings": 128}}
        self.source_path.write_bytes(experiment._canonical(self.source_record) + b"\n")
        self.prepared_path.write_bytes(experiment._canonical(self.prepared_record) + b"\n")
        self.provider, self.tokenizer = _Provider(), _Tokenizer()
        self.provider_factory = Mock(return_value=self.provider)
        for name, value in (("_verify_source", lambda path: deepcopy(self.source_record)),
                            ("_verify_preparation", lambda path: deepcopy(self.prepared_record)),
                            ("_runtime", lambda prepared: (torch, self.tokenizer, self.provider_factory))):
            patched = patch.object(experiment, name, value)
            patched.start()
            self.addCleanup(patched.stop)
        self.recipe = {"train_rows_per_family": 1, "dev_rows_per_family": 3,
                       "style_rows_per_dev_family": 1, "width": 6, "steps": 4,
                       "seed": 41, "learning_rate": 0.001, "weight_decay": 0.01}
        self.plan_path, self.cache_path = self.root / "plan.json", self.root / "cache"

    def _plan(self, **kwargs):
        return experiment.create_plan(source_admission=self.source_path, preparation_receipt=self.prepared_path,
                                      output_plan=kwargs.pop("output", self.plan_path),
                                      recipe=kwargs.pop("recipe", self.recipe), **kwargs)

    def _run(self, output="run"):
        return experiment.run_experiment(plan_path=self.plan_path, output_directory=self.root / output,
                                         cache_directory=self.cache_path)

    def test_deterministic_plan_full_raw_roles_no_class_ids_and_lexical_metric_rule(self):
        plan = self._plan()
        plan2 = self._plan(output=self.root / "plan2.json")
        self.assertEqual(plan, plan2)
        self.assertEqual(len(plan["examples"]), 5)
        self.assertEqual(plan["original_source_sentence_only_overlap"], 1)
        dev = [example for example in plan["examples"] if example["split"] == "dev"]
        self.assertEqual(sum(example["sentence_unseen_in_all_train"] for example in dev), 2)
        for example in plan["examples"]:
            text = example["source"]["text"]
            self.assertIn("Head entity:", text)
            self.assertIn("Tail entity:", text)
            self.assertFalse(re.search(r"\b[PQ]\d+\b", text))
            self.assertNotIn("Qhead-fixture", text)
            self.assertNotIn(example["id_metadata"], text)
        self.assertFalse(plan["source_counts_define_capability_ceiling"])

    def test_full_cpu_fixture_actual_updates_controls_resume_export_and_immutable_cache_reuse(self):
        plan = self._plan()
        def load_after_budget():
            budget = experiment._read(self.root / "run" / "feature-budget.json")
            experiment._verify_seal(budget)
            self.assertIs(budget["model_loaded"], False)
            self.assertEqual(budget["complete_unique_inputs"], plan["unique_complete_feature_inputs"])
            self.assertIsNone(budget["cpu_peak_memory_bytes"])
            self.assertEqual(len(budget["inputs"]), plan["unique_complete_feature_inputs"])
            self.assertTrue(all(len(item["raw_token_binding_sha256"]) == 64 for item in budget["inputs"]))
            return self.provider
        self.provider_factory.side_effect = load_after_budget
        receipt = self._run()
        self.assertEqual(receipt["state"], "PUBLIC_SEMANTIC_EXPERIMENT_UNQUALIFIED")
        self.assertFalse(receipt["n0_approved"])
        self.assertFalse(receipt["personality_qualified"])
        self.assertFalse(receipt["upstream_gradient"])
        self.assertFalse(receipt["final_payload_opened"])
        self.assertEqual(len(receipt["training_loss_history"]), self.recipe["steps"])
        self.assertTrue(receipt["checkpoint"]["optimizer_rng_resume_exact"])
        self.assertTrue(receipt["export"]["reload_exact"])
        self.assertEqual(receipt["learned"]["dev"]["rows"], 3)
        self.assertEqual(receipt["learned"]["sentence_unseen_dev"]["rows"], 2)
        self.assertEqual(receipt["candidate_permutation"]["learned"]["max_aligned_logit_difference"], 0)
        self.assertEqual(receipt["inherited_helper_wording_sensitivity"]["learned"]["variants"]["helpful"]["rows"], 1)
        self.assertEqual(self.provider.calls, plan["unique_complete_feature_inputs"])
        self.provider_factory.assert_called_once()
        budget_path = Path(receipt["pre_forward_feature_budget"]["path"])
        self.assertEqual(experiment._digest(budget_path), receipt["pre_forward_feature_budget"]["sha256"])
        self.assertEqual(receipt["pre_forward_feature_budget"]["estimated_logical_feature_cache_bytes"],
                         sum(record["logical_feature_bytes"] for record in receipt["cache"]))
        before = {path.name: experiment._digest(path) for path in self.cache_path.iterdir()}
        self.provider_factory.reset_mock()
        self.provider_factory.side_effect = AssertionError("must reuse immutable public cache")
        repeated = self._run("second-run")
        self.assertEqual(repeated["learned"], receipt["learned"])
        self.assertEqual(repeated["training_loss_history"], receipt["training_loss_history"])
        self.provider_factory.assert_not_called()
        self.assertEqual(before, {path.name: experiment._digest(path) for path in self.cache_path.iterdir()})
        self.assertTrue(all(not record["created"] for record in repeated["cache"]))

    def test_plan_target_or_input_forgery_rejected_before_runtime(self):
        plan = self._plan()
        plan.pop("receipt_sha256")
        plan["examples"][0]["target_index_metadata"] = 99
        self.plan_path.write_bytes(experiment._canonical(experiment._seal(plan)) + b"\n")
        with patch.object(experiment, "_runtime", side_effect=AssertionError("runtime before targets")), \
                self.assertRaisesRegex(experiment.ExperimentError, "subset/targets"):
            self._run()

    def test_preparation_file_and_code_change_stop_before_runtime(self):
        self._plan()
        self.prepared_path.write_bytes(self.prepared_path.read_bytes() + b" ")
        with patch.object(experiment, "_runtime", side_effect=AssertionError("runtime")), \
                self.assertRaisesRegex(experiment.ExperimentError, "receipt file changed"):
            self._run()
        self.prepared_path.write_bytes(experiment._canonical(self.prepared_record) + b"\n")
        with patch.object(experiment, "_code", return_value=[]), \
                self.assertRaisesRegex(experiment.ExperimentError, "plan/code"):
            self._run()

    def test_no_target_metadata_may_enter_source_text(self):
        self.rows[0]["head"]["type"] = "Ada"
        self.row_path.write_bytes(b"".join(experiment._canonical(row) + b"\n" for row in self.rows))
        self.source_record["files"][0]["sha256"] = experiment._digest(self.row_path)
        with self.assertRaisesRegex(experiment.ExperimentError, "forbidden target/opaque"):
            self._plan(recipe={**self.recipe, "train_rows_per_family": 3})

    def test_raw_tokenizer_truncation_and_actual_context_budget_rejected(self):
        self._plan()
        self.tokenizer.truncate_offsets_only = True
        with self.assertRaisesRegex(experiment.ExperimentError, "retokenization"):
            self._run("bad-tokenizer")
        self.tokenizer.truncate_offsets_only = False
        self.prepared_record["model_geometry"]["max_position_embeddings"] = 1
        self.prepared_path.write_bytes(experiment._canonical(self.prepared_record) + b"\n")
        self._plan(output=self.root / "small-budget-plan.json")
        with self.assertRaisesRegex(experiment.ExperimentError, "context"):
            experiment.run_experiment(plan_path=self.root / "small-budget-plan.json",
                                      output_directory=self.root / "over-budget", cache_directory=self.root / "other-cache")
        self.assertEqual(self.provider.calls, 0)

    def test_cache_tensor_ids_cannot_be_resealed_to_different_raw_source(self):
        self._plan()
        receipt = self._run()
        record = receipt["cache"][0]
        tensor_path, metadata_path = Path(record["tensor_path"]), Path(record["metadata_path"])
        payload = torch.load(tensor_path, weights_only=True)
        payload["input_ids"][0] += 1
        with tensor_path.open("wb") as stream:
            torch.save(payload, stream)
        metadata = experiment._read(metadata_path)
        metadata.pop("receipt_sha256")
        metadata["tensor_sha256"] = experiment._digest(tensor_path)
        metadata_path.write_bytes(experiment._canonical(experiment._seal(metadata)) + b"\n")
        with self.assertRaisesRegex(experiment.ExperimentError, "cached IDs/masks"):
            self._run("poisoned-cache")

    def test_checkpoint_rejects_wrong_bytes_binding_and_export(self):
        self._plan()
        receipt = self._run()
        model = SemanticReadout(state_count=3, hidden_size=8, width=6)
        optimizer = torch.optim.AdamW(model.parameters())
        sampler = torch.Generator()
        checkpoint = Path(receipt["checkpoint"]["path"])
        with self.assertRaisesRegex(experiment.ExperimentError, "external pin"):
            experiment.restore_checkpoint(checkpoint, model=model, optimizer=optimizer, sampler=sampler,
                                          binding=receipt["binding"], expected_file_sha256="0" * 64, torch=torch)
        with self.assertRaisesRegex(experiment.ExperimentError, "binding differs"):
            experiment.restore_checkpoint(checkpoint, model=model, optimizer=optimizer, sampler=sampler,
                                          binding={"wrong": True}, expected_file_sha256=experiment._digest(checkpoint), torch=torch)
        with self.assertRaisesRegex(experiment.ExperimentError, "export binding"):
            experiment.load_semantic_export(Path(receipt["export"]["path"]), binding={"wrong": True},
                                             expected_file_sha256=receipt["export"]["sha256"], torch=torch)

    def test_dev_rejected_before_any_optimizer_update(self):
        from tests.personality.test_gemma_n0_semantic_readout import bank
        model = SemanticReadout(state_count=3, hidden_size=8, width=6)
        original = deepcopy(model.state_dict())
        with self.assertRaisesRegex(experiment.ExperimentError, "DEV cannot"):
            experiment._train_steps(model, torch.optim.AdamW(model.parameters()), torch.Generator(),
                                     [{"split": "dev", "source": bank(source=True), "candidates": [bank()], "target": 0}],
                                     steps=1, torch=torch)
        self.assertTrue(experiment._equal(original, model.state_dict(), torch))

    def test_source_model_output_boundaries_recipe_and_empty_lexical_subset(self):
        with self.assertRaisesRegex(experiment.ExperimentError, "outside source"):
            self._plan(output=self.data / "plan.json")
        with self.assertRaisesRegex(experiment.ExperimentError, "at least two"):
            self._plan(recipe={**self.recipe, "steps": 1})
        self.assertEqual(experiment._evaluate(None, [], torch)["rows"], 0)
        self.assertIsNone(experiment._evaluate(None, [], torch)["accuracy"])

    def test_cli_plan_does_not_load_runtime_and_declares_operating_subset(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(experiment, "_runtime", side_effect=AssertionError("plan loads model")):
            self.assertEqual(cli.main(["plan", "--source-admission", str(self.source_path),
                                       "--preparation", str(self.prepared_path), "--output", str(self.plan_path),
                                       "--train-rows-per-family", "1", "--dev-rows-per-family", "2",
                                       "--steps", "4", "--width", "6"]), 0)
        self.assertFalse(json.loads(output.getvalue())["n0_approved"])


if __name__ == "__main__":
    unittest.main()
