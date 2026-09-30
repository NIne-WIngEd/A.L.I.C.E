"""CPU-only tests for sealed V1 specialist evaluation; no 12B checkpoint."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning import OUTPUT_SCHEMA
from scripts.mfm.evaluate_base_behavior import emit_inputs, load_diagnostic, score_case
from scripts.mfm import run_v1_formation_specialist as inference
from scripts.mfm import train_v1_formation_specialist as training

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_diagnostic()

    def test_public_inference_input_replays_exact_training_source_encoding(self):
        case = self.cases["document-memory-injection"]
        row = next(item for item in emit_inputs(self.cases)
                   if item["case_id"] == "document-memory-injection")
        case_id, context, opened = inference.read_input(row)
        self.assertEqual(context, case[0].context)
        self.assertEqual(opened, case[1])
        calls = []

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return {"input_ids": _tensor([[1, 2, 3]]),
                        "attention_mask": _tensor([[1, 1, 1]])}

        processor = Processor()
        fake_example = types.SimpleNamespace(context=context, opened_sources=opened,
                                             case_id=case_id)
        a = training._source_batch(processor, fake_example, 16)
        b = training.source_batch(processor, context, opened, 16, case_id=case_id)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(a["input_ids"].tolist(), b["input_ids"].tolist())
        self.assertNotIn("<|image|>", str(calls[0][0]))

    def test_source_tamper_and_context_tamper_fail_closed(self):
        row = next(item for item in emit_inputs(self.cases)
                   if item["case_id"] == "neutral-owner-event")
        row["context_digest"] = "0" * 64
        with self.assertRaisesRegex(CognitiveKernelContractError, "context digest"):
            inference.read_input(row)
        row["context_digest"] = self.cases["neutral-owner-event"][0].context.content_digest()
        row["opened_sources"][0]["payload_base64"] = "dGFtcGVy"
        with self.assertRaisesRegex(CognitiveKernelContractError, "source digest"):
            inference.read_input(row)

    def test_runtime_refuses_private_input_without_network_isolation(self):
        args = types.SimpleNamespace(input_jsonl=Path("/private/not-opened.jsonl"))
        with patch.object(training.socket, "if_nameindex", return_value=[
                (1, "lo"), (2, "eth0")]):
            with self.assertRaisesRegex(CognitiveKernelContractError, "loopback-only"):
                inference.run(args)


def _tensor(value):
    if torch is None:
        raise unittest.SkipTest("CPU PyTorch not installed")
    return torch.tensor(value)


@unittest.skipIf(torch is None, "CPU PyTorch not installed")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_diagnostic()
        self.gold, self.opened, _ = self.cases["neutral-owner-event"]
        self.context = self.gold.context

    def _generation(self, raw: str, *, max_new_tokens: int | None = None):
        class FakeDecoder:
            config = types.SimpleNamespace(start_token_id=1, end_token_id=2,
                                           max_target_tokens=256)

            def parameters(self):
                yield torch.zeros(1)

            def next_token_logits(self, *, base_states, source_mask, input_ids):
                index = input_ids.shape[1] - 1
                token = ord(raw[index]) if index < len(raw) else 2
                logits = torch.full((1, 128), -1.0)
                logits[0, token] = 1.0
                return logits

        class FakeTokenizer:
            def decode(self, ids, **kwargs):
                return "".join(chr(item) for item in ids)

        processor = types.SimpleNamespace(tokenizer=FakeTokenizer())
        base = types.SimpleNamespace(model=lambda **kwargs: types.SimpleNamespace(
            last_hidden_state=torch.ones((1, 3, 4))))
        with patch.object(training, "source_batch", return_value={
                "input_ids": torch.tensor([[3, 4, 5]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long)}):
            result = inference.generate_case(
                processor=processor, base=base, specialist=FakeDecoder(),
                context=self.context, opened_sources=self.opened, case_id=self.gold.case_id,
                component_sha256="a" * 64, inference_run_id="cpu-test",
                max_source_tokens=16,
                max_new_tokens=max_new_tokens or len(raw) + 1)
        return result

    def test_valid_canonical_output_and_raw_evaluation(self):
        raw = json.dumps({"schema": OUTPUT_SCHEMA, "proposals": [], "dispositions": []},
                         separators=(",", ":"))
        generated = self._generation(raw)
        self.assertEqual(generated["validation_status"], "grounded_proposal_only")
        self.assertEqual(generated["output_text"], raw)
        self.assertTrue(generated["eos_observed"])
        score = score_case(self.gold, self.opened, {
            **generated, "case_id": self.gold.case_id, "status": "generated",
            "context_digest": self.context.content_digest(),
            "model_artifact_digest": "a" * 64,
            "processed_modalities": ["text"]})
        self.assertFalse(score["critical_failures"])

    def test_invalid_raw_and_missing_eos_remain_failures(self):
        for raw in ('{"schema":"wrong","proposals":[],"dispositions":[]}',
                    'not-json'):
            result = self._generation(raw)
            self.assertEqual(result["output_text"], raw)
            self.assertEqual(result["validation_status"], "invalid")
            self.assertIsNotNone(result["validation_error"])
        truncated = self._generation("{" + "x" * 20, max_new_tokens=4)
        self.assertEqual(truncated["output_text"], "{xxx")
        self.assertFalse(truncated["eos_observed"])
        self.assertEqual(truncated["validation_status"], "invalid")

    def test_real_decoder_only_projects_last_position(self):
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
        config = SpecialistConfig(4, 31, 8, 1, 2, 8, 0, 1, 2, 2)
        model = FormationSpecialist(config).eval()
        source = torch.ones((1, 3, 4))
        mask = torch.ones((1, 3), dtype=torch.long)
        ids = torch.tensor([[1, 3, 4]])
        with torch.inference_mode():
            full = model(base_states=source, source_mask=mask, input_ids=ids)
            last = model.next_token_logits(base_states=source, source_mask=mask,
                                           input_ids=ids)
        torch.testing.assert_close(last, full[:, -1])

    def test_sealed_lineage_rejects_modified_component_and_probe(self):
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
        config = SpecialistConfig(4, 31, 8, 1, 2, 8, 0, 1, 2, 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specialist = FormationSpecialist(config)
            run_data = {"schema": training.RUN_SCHEMA, "probe_only": False,
                        "preflight_sha256": None,
                        "prepared_base_receipt_sha256": "a" * 64,
                        "corpus_sha256": "b" * 64,
                        "specialist_config": config.record()}
            prepared = {"receipt_sha256": "a" * 64, "schema": training.CLONE_SCHEMA,
                        "files": [{"path": "model.safetensors",
                                   "sha256": training.SOURCE_WEIGHT_SHA256}]}
            preflight = {"schema": training.PREFLIGHT_SCHEMA,
                         "objective": training.OBJECTIVE_VERSION,
                         "corpus_sha256": "b" * 64,
                         "prepared_base_receipt_sha256": "a" * 64,
                         "prepared_base_weight_sha256": training.SOURCE_WEIGHT_SHA256,
                         "prepared_base_parent_weight_sha256": training.SOURCE_WEIGHT_SHA256,
                         "prepared_base_kind": "licensed-verified-mfm-role-clone",
                         "foundation_commit": training.FOUNDATION_COMMIT,
                         "foundation_verifier_sha256": training.FOUNDATION_VERIFIER_SHA256,
                         "foundation_inventory_sha256": training.FOUNDATION_INVENTORY_SHA256,
                         "source_template_sha256": sha256(training.SOURCE_TEMPLATE.encode()).hexdigest(),
                         "source_builder_sha256": training._digest(Path(
                             __import__(training.formation_context_media_messages.__module__,
                                        fromlist=["__file__"]).__file__)),
                         "trainer_sha256": training._digest(Path(training.__file__)),
                         "max_source_tokens": 32,
                         "max_target_tokens": config.max_target_tokens,
                         "specialist_heads": config.heads,
                         "specialist_layers": config.layers}
            preflight = training._write_new(root / "preflight.json", preflight)
            run_data["preflight_sha256"] = preflight["record_sha256"]
            run = training._write_new(root / "run.json", run_data)
            training._seed_control(root, specialist, run["record_sha256"], config)
            save_file(specialist.state_dict(), str(root / "formation-specialist.safetensors"))
            component = {"schema": training.ARTIFACT_SCHEMA,
                         "formation_component_sha256": training._digest(root / "formation-specialist.safetensors"),
                         "prepared_base_sha256": training.SOURCE_WEIGHT_SHA256,
                         "prepared_base_parent_sha256": training.SOURCE_WEIGHT_SHA256,
                         "prepared_base_receipt_sha256": "a" * 64,
                         "prepared_base_kind": "licensed-verified-mfm-role-clone",
                         "run_manifest_sha256": run["record_sha256"],
                         "seed_control_sha256": training._digest(root / "seed-control.safetensors"),
                         "specialist_config": config.record(), "probe_only": False,
                         "optimizer_steps": 1}
            training._write_new(root / "formation-component.json", component)
            with patch.object(training, "_prepared_base", return_value=prepared):
                for role in ("trained", "seeded-untrained"):
                    lineage = inference.verify_artifacts(root, root, root / "clone.json",
                                                         root / "preflight.json", control=role)
                    self.assertTrue(lineage[3].is_file())
                (root / "formation-component.json").unlink()
                training._write_new(root / "formation-component.json",
                                    {**component, "probe_only": True})
                with self.assertRaisesRegex(CognitiveKernelContractError, "hardware probe"):
                    inference.verify_artifacts(root, root, root / "clone.json",
                                               root / "preflight.json", control="trained")
                (root / "formation-component.json").unlink()
                training._write_new(root / "formation-component.json", component)
                (root / "formation-specialist.safetensors").write_bytes(b"tamper")
                with self.assertRaisesRegex(CognitiveKernelContractError, "weight bytes"):
                    inference.verify_artifacts(root, root, root / "clone.json",
                                               root / "preflight.json", control="trained")
                save_file(specialist.state_dict(), str(root / "formation-specialist.safetensors"))
                (root / "preflight.json").unlink()
                preflight["corpus_sha256"] = "c" * 64
                training._write_new(root / "preflight.json", {
                    key: value for key, value in preflight.items() if key != "record_sha256"})
                with self.assertRaisesRegex(CognitiveKernelContractError, "lineage differs"):
                    inference.verify_artifacts(root, root, root / "clone.json",
                                               root / "preflight.json", control="trained")


if __name__ == "__main__":
    unittest.main()
