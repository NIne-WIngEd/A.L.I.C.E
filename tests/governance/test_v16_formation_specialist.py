"""CPU contract checks for the separate v1.6 specialist; no 12B run."""

from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import (
    learning_example_v16_from_record, supervised_output_record_v16,
)
from scripts.mfm import train_v16_formation_specialist as trainer
from scripts.mfm import run_v16_formation_specialist as runner
from tests.governance.test_formation_learning_v16 import case_record
from tests.governance.test_formation_semantics_v16 import SOURCE


try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


class V16AdmissionTests(unittest.TestCase):
    def test_authored_seed_keeps_development_out_of_gradient_inputs(self):
        from scripts.mfm.build_v16_authoring_seed import render

        raw = render()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authored.jsonl"
            path.write_bytes(raw)
            train, development = trainer._public_examples(
                path, sha256(raw).hexdigest(), "owner-directed-mfm-v16-synthetic")
        frozen_roster = [json.loads(line) for line in raw.splitlines()]
        self.assertGreater(len(train), 0)
        self.assertGreater(len(development), 0)
        self.assertEqual(len(train), sum(row["split"] == "train" for row in frozen_roster))
        self.assertEqual(len(development), sum(row["split"] == "development" for row in frozen_roster))
        self.assertEqual({example.split for example in train}, {"train"})
        self.assertEqual({example.split for example in development}, {"development"})

    def test_public_cpu_checks_require_exact_v16_bytes_and_ten_labels(self):
        row = case_record()
        row["authorization_id"] = "owner-approved-public-test"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public.jsonl"
            path.write_text(json.dumps(row) + "\n")
            actual = sha256(path.read_bytes()).hexdigest()
            self.assertEqual(tuple(map(len, trainer._public_examples(
                path, actual, "owner-approved-public-test"))), (1, 0))
            row["target"]["adjudications"]["correction"] = "unknown"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "unknown adjudication"):
                trainer._public_examples(path, sha256(path.read_bytes()).hexdigest(),
                                         "owner-approved-public-test")
            row["target"]["adjudications"]["correction"] = "negative"
            row["schema"] = "mfm-curriculum-case-v1"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "wrong v1.6 schema"):
                trainer._public_examples(path, sha256(path.read_bytes()).hexdigest(),
                                         "owner-approved-public-test")

    def test_full_fit_blocks_before_private_data_or_gpu(self):
        args = types.SimpleNamespace(mode="train", probe_only=False,
                                     resume_checkpoint=None,
                                     public_synthetic_curriculum=None,
                                     max_source_tokens=8, max_target_tokens=8,
                                     specialist_width=8, specialist_layers=1,
                                     specialist_heads=2, logit_chunk_tokens=1,
                                     epochs=1, gradient_accumulation=1,
                                     save_every_steps=1, learning_rate=1e-4,
                                     max_cross_attention_pairs=8)
        with patch.object(trainer, "arguments", return_value=args), \
                patch.object(trainer, "_examples", side_effect=AssertionError("private open")), \
                self.assertRaisesRegex(CognitiveKernelContractError,
                                        "independent adjudication"):
            trainer.main()

    def test_admitted_private_preflight_checks_isolation_before_manifest_open(self):
        args = types.SimpleNamespace(mode="data-preflight", probe_only=False,
                                     resume_checkpoint=None,
                                     public_synthetic_curriculum=None,
                                     admitted_manifest=Path("/private/never-open.json"),
                                     max_source_tokens=8, max_target_tokens=8,
                                     specialist_width=8, specialist_layers=1,
                                     specialist_heads=2, logit_chunk_tokens=1,
                                     epochs=1, gradient_accumulation=1,
                                     save_every_steps=1, learning_rate=1e-4,
                                     max_cross_attention_pairs=8)
        with patch.object(trainer, "arguments", return_value=args), \
                patch.object(trainer.shared.socket, "if_nameindex",
                             return_value=[(1, "lo"), (2, "eth0")]), \
                patch.object(trainer, "_examples", side_effect=AssertionError("private open")), \
                self.assertRaisesRegex(CognitiveKernelContractError, "loopback-only"):
            trainer.main()


@unittest.skipIf(torch is None, "CPU PyTorch not installed")
class V16InferenceTests(unittest.TestCase):
    def test_versioned_source_context_is_observed_and_media_markers_escaped(self):
        example = learning_example_v16_from_record(case_record())
        calls = []

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                calls.append(messages)
                return {"input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.ones((1, 3), dtype=torch.long)}

        source = trainer.source_batch_v16(
            Processor(), example.context, example.opened_sources, 64,
            case_id=example.case_id)
        self.assertEqual(source["input_ids"].shape[1], 3)
        instruction = calls[0][0]["content"][0]["text"]
        self.assertIn("mfm-grounded-full-role-objective-v1.6", instruction)
        self.assertIn("source_sensitivities", instruction)
        self.assertIn("mission", instruction)
        self.assertNotIn("mfm-grounded-multimodal-disposition-objective-v1", instruction)
        self.assertNotIn("<|image|>", instruction)
        with self.assertRaisesRegex(CognitiveKernelContractError, "source order"):
            trainer.source_batch_v16(Processor(), example.context, (), 64)

    def test_target_tokens_come_from_v16_serializer_and_reject_unknown(self):
        example = learning_example_v16_from_record(case_record())

        class Tokenizer:
            bos_token_id = 1
            eos_token_id = 2

            def encode(self, text, **kwargs):
                self.output = json.loads(text)
                return [3, 4]

        tokenizer = Tokenizer()
        self.assertEqual(trainer._target_ids(tokenizer, example, 4),
                         ([1, 3, 4], [3, 4, 2]))
        self.assertEqual(tokenizer.output, supervised_output_record_v16(example))
        row = case_record()
        row["target"]["adjudications"]["outcome"] = "unknown"
        with self.assertRaisesRegex(CognitiveKernelContractError,
                                    "unknown adjudication"):
            trainer._target_ids(tokenizer, learning_example_v16_from_record(row), 4)

    def test_input_context_and_source_hash_cannot_be_swapped(self):
        example = learning_example_v16_from_record(case_record())
        row = {"case_id": example.case_id,
               "context": example.context.record(),
               "context_digest": example.context.content_digest(),
               "opened_sources": [{"ref_id": "source-1",
                                   "payload_base64": b64encode(SOURCE).decode()}]}
        self.assertEqual(runner.read_input(row)[1], example.context)
        row["context_digest"] = "0" * 64
        with self.assertRaisesRegex(CognitiveKernelContractError,
                                    "context digest"):
            runner.read_input(row)
        row["context_digest"] = example.context.content_digest()
        row["opened_sources"][0]["payload_base64"] = b64encode(b"tamper").decode()
        with self.assertRaisesRegex(CognitiveKernelContractError,
                                    "source digest"):
            runner.read_input(row)
        row["opened_sources"][0]["payload_base64"] = b64encode(SOURCE).decode() + "="
        with self.assertRaisesRegex(CognitiveKernelContractError,
                                    "source encoding"):
            runner.read_input(row)

    def test_generated_output_uses_v16_grounding_and_keeps_invalid_raw(self):
        example = learning_example_v16_from_record(case_record())
        raw = json.dumps(supervised_output_record_v16(example),
                         ensure_ascii=True, separators=(",", ":"))

        class Decoder:
            config = types.SimpleNamespace(start_token_id=1, end_token_id=2,
                                           max_target_tokens=8192)

            def parameters(self):
                yield torch.zeros(1)

            def next_token_logits(self, *, base_states, source_mask, input_ids):
                index = input_ids.shape[1] - 1
                token = ord(self.raw[index]) if index < len(self.raw) else 2
                logits = torch.full((1, 128), -1.0)
                logits[0, token] = 1.0
                return logits

        class Processor:
            tokenizer = types.SimpleNamespace(
                decode=lambda ids, **kwargs: "".join(chr(item) for item in ids))

        base = types.SimpleNamespace(model=lambda **kwargs: types.SimpleNamespace(
            last_hidden_state=torch.ones((1, 3, 4))))
        decoder = Decoder()
        with patch.object(trainer, "source_batch_v16", return_value={
                "input_ids": torch.tensor([[3, 4, 5]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long)}):
            decoder.raw = raw
            valid = runner.generate_case(
                processor=Processor(), base=base, specialist=decoder,
                context=example.context, opened_sources=example.opened_sources,
                case_id=example.case_id, component_sha256="a" * 64,
                inference_run_id="test-v16", max_source_tokens=16,
                max_new_tokens=len(raw) + 1)
            self.assertEqual(valid["validation_status"], "grounded_probe_proposal_only")
            decoder.raw = raw.replace("mfm-formation-output-v1.6",
                                      "mfm-formation-output-v1", 1)
            invalid = runner.generate_case(
                processor=Processor(), base=base, specialist=decoder,
                context=example.context, opened_sources=example.opened_sources,
                case_id=example.case_id, component_sha256="a" * 64,
                inference_run_id="test-v16-invalid", max_source_tokens=16,
                max_new_tokens=len(decoder.raw) + 1)
            self.assertEqual(invalid["validation_status"], "invalid")
            self.assertEqual(invalid["raw_output_text"], decoder.raw)

    def test_sealed_v16_probe_and_control_reject_v15_receipt(self):
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig

        config = SpecialistConfig(4, 31, 8, 1, 2, 8, 0, 1, 2, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specialist = FormationSpecialist(config)
            prepared = {"receipt_sha256": "a" * 64, "schema": trainer.shared.CLONE_SCHEMA,
                        "files": [{"path": "model.safetensors",
                                   "sha256": trainer.shared.SOURCE_WEIGHT_SHA256}]}
            preflight = {
                "schema": trainer.PREFLIGHT_SCHEMA,
                "objective": trainer.OBJECTIVE_VERSION_V16,
                "output_schema": "mfm-formation-output-v1.6",
                "context_schema": "mfm-formation-context-v1.6",
                "prepared_base_receipt_sha256": "a" * 64,
                "prepared_base_weight_sha256": trainer.shared.SOURCE_WEIGHT_SHA256,
                "prepared_base_parent_weight_sha256": trainer.shared.SOURCE_WEIGHT_SHA256,
                "prepared_base_kind": "licensed-verified-mfm-role-clone",
                "foundation_commit": trainer.shared.FOUNDATION_COMMIT,
                "foundation_verifier_sha256": trainer.shared.FOUNDATION_VERIFIER_SHA256,
                "foundation_inventory_sha256": trainer.shared.FOUNDATION_INVENTORY_SHA256,
                "source_template_sha256": sha256(trainer.shared.SOURCE_TEMPLATE.encode()).hexdigest(),
                "source_instruction_sha256": sha256(trainer.SOURCE_INSTRUCTION.encode()).hexdigest(),
                "source_builder_sha256": trainer.shared._digest(Path(
                    __import__(trainer.formation_context_media_messages.__module__,
                               fromlist=["__file__"]).__file__)),
                "codec_sha256": trainer._codec_sha256(),
                "semantics_sha256": trainer._semantics_sha256(),
                "decoder_implementation_sha256": trainer._decoder_sha256(),
                "trainer_sha256": trainer.shared._digest(Path(trainer.__file__)),
                "max_source_tokens": 64,
                "max_target_tokens": config.max_target_tokens,
                "specialist_heads": config.heads,
                "specialist_layers": config.layers,
                "corpus_sha256": "b" * 64,
                "transformers_version": "test-version",
            }
            preflight = trainer.shared._write_new(root / "preflight.json", preflight)
            run = trainer.shared._write_new(root / "run.json", {
                "schema": trainer.RUN_SCHEMA,
                "objective": trainer.OBJECTIVE_VERSION_V16,
                "preflight_sha256": preflight["record_sha256"],
                "prepared_base_receipt_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
                "specialist_config": config.record(),
                "transformers_version": "test-version",
                "probe_only": True,
                "qualified_for_product": False,
            })
            trainer._seed_control(root, specialist, run["record_sha256"], config)
            with torch.no_grad():
                specialist.source_projection.weight[0, 0] += 0.25
            save_file(specialist.state_dict(), str(root / "formation-specialist.safetensors"))
            component = {
                "schema": trainer.ARTIFACT_SCHEMA,
                "objective": trainer.OBJECTIVE_VERSION_V16,
                "formation_component_sha256": trainer.shared._digest(
                    root / "formation-specialist.safetensors"),
                "prepared_base_sha256": trainer.shared.SOURCE_WEIGHT_SHA256,
                "prepared_base_parent_sha256": trainer.shared.SOURCE_WEIGHT_SHA256,
                "prepared_base_receipt_sha256": "a" * 64,
                "prepared_base_kind": "licensed-verified-mfm-role-clone",
                "run_manifest_sha256": run["record_sha256"],
                "seed_control_sha256": trainer.shared._digest(
                    root / "seed-control.safetensors"),
                "specialist_config": config.record(), "probe_only": True,
                "optimizer_steps": 1, "qualified_for_product": False,
            }
            trainer.shared._write_new(root / "formation-component.json", component)
            with patch.object(trainer.shared, "_prepared_base", return_value=prepared):
                for control in ("trained", "seeded-untrained"):
                    selected = runner.verify_artifacts(root, root, root / "clone.json",
                                                       root / "preflight.json", control=control)[3]
                    self.assertTrue(selected.is_file())
                (root / "preflight.json").unlink()
                trainer.shared._write_new(root / "preflight.json", {
                    **{key: value for key, value in preflight.items()
                       if key != "record_sha256"},
                    "objective": "mfm-grounded-disposition-objective-v1"})
                with self.assertRaisesRegex(CognitiveKernelContractError,
                                            "objective lineage"):
                    runner.verify_artifacts(root, root, root / "clone.json",
                                            root / "preflight.json", control="trained")


if __name__ == "__main__":
    unittest.main()
