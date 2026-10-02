"""Features-only boundary tests with actual small CPU tensors, no Gemma load.

They test custody admission ordering, source shape preservation and frozen
gradients. They are not behavioral qualification of the 12B checkpoint.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src.alice_personality.gemma_n0 import backbone, preparation
from src.alice_foundation import gemma4_v1 as foundation

try:
    import torch
except ImportError:
    torch = None


class LazyImportTests(unittest.TestCase):
    def test_module_import_does_not_require_model_packages(self):
        root = Path(__file__).resolve().parents[2]
        code = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise RuntimeError('model package imported during custody-only import')
sys.meta_path.insert(0, BlockModels())
from src.alice_personality.gemma_n0 import backbone
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root)
        result = subprocess.run([sys.executable, "-c", code], cwd=root,
                                env=environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_preparation_never_imports_or_loads_model_packages(self):
        with patch.object(preparation, "verify_prepared", side_effect=preparation.PreparationError("wrong role")), \
                patch.object(backbone, "import_module") as imports:
            with self.assertRaises(preparation.PreparationError):
                backbone.load_prepared_gemma_n0("unused-receipt.json")
        imports.assert_not_called()

    def test_dtype_override_must_match_before_model_import(self):
        receipt = {"runtime": {"dtype": "bfloat16"}}
        with patch.object(preparation, "verify_prepared", return_value=receipt), \
                patch.object(backbone, "import_module") as imports:
            with self.assertRaises(backbone.BackboneError):
                backbone.load_prepared_gemma_n0("unused.json", dtype="float32")
        imports.assert_not_called()


@unittest.skipIf(torch is None, "install CPU Torch to verify frozen representations and gradients")
class FrozenRepresentationTests(unittest.TestCase):
    def setUp(self):
        class Representation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)
                self.dropout = torch.nn.Dropout(0.9)
                self.inputs = None
                self.output_override = None

            def forward(self, **inputs):
                self.inputs = inputs
                if self.output_override is not None:
                    return types.SimpleNamespace(last_hidden_state=self.output_override)
                states = self.dropout(self.embedding(inputs["input_ids"]))
                return types.SimpleNamespace(last_hidden_state=states)

        self.representation = Representation()
        self.n0 = backbone._FrozenGemmaN0(self.representation, torch=torch,
                                        hidden_size=4, max_source_tokens=5, device="cpu")
        self.ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def test_complete_token_positions_and_mask_are_detached_and_preserved(self):
        features = self.n0(input_ids=self.ids, attention_mask=self.mask)
        self.assertEqual(features.hidden_states.shape, (2, 3, 4))
        self.assertTrue(torch.equal(features.attention_mask, self.mask.bool()))
        self.assertFalse(features.hidden_states.requires_grad)
        self.assertIsNone(features.hidden_states.grad_fn)
        self.assertFalse(features.attention_mask.requires_grad)
        self.assertTrue(torch.equal(self.representation.inputs["input_ids"], self.ids))
        self.assertIs(self.representation.inputs["use_cache"], False)
        self.assertIs(self.representation.inputs["return_dict"], True)
        self.assertEqual(self.n0.device, torch.device("cpu"))

    def test_caller_train_and_external_toggle_cannot_create_base_gradient_path(self):
        self.n0.train()
        self.assertFalse(self.n0.training)
        self.assertFalse(self.representation.training)
        self.representation.requires_grad_(True).train()
        features = self.n0(input_ids=self.ids, attention_mask=self.mask)
        self.assertFalse(self.representation.training)
        self.assertTrue(all(not p.requires_grad for p in self.representation.parameters()))
        downstream = torch.nn.Linear(4, 2)
        downstream(features.hidden_states).sum().backward()
        self.assertIsNotNone(downstream.weight.grad)
        self.assertTrue(torch.isfinite(downstream.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in self.representation.parameters()))

    def test_eval_is_deterministic_under_same_source(self):
        first = self.n0(input_ids=self.ids, attention_mask=self.mask).hidden_states
        second = self.n0(input_ids=self.ids, attention_mask=self.mask).hidden_states
        self.assertTrue(torch.equal(first, second))

    def test_source_tensor_passthrough_keeps_exact_media_values(self):
        media = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        self.n0(input_ids=self.ids, attention_mask=self.mask, pixel_values=media)
        self.assertTrue(torch.equal(self.representation.inputs["pixel_values"], media))

    def test_only_known_processor_media_names_pass_the_feature_boundary(self):
        media = torch.tensor([1])
        accepted = ("pixel_values", "pixel_values_videos", "input_features", "input_features_mask",
                    "mm_token_type_ids", "image_position_ids", "video_position_ids")
        for name in accepted:
            with self.subTest(name=name):
                self.n0(input_ids=self.ids, attention_mask=self.mask, **{name: media})
                self.assertTrue(torch.equal(self.representation.inputs[name], media))
        self.representation.inputs = None
        for name in ("position_ids", "cache_position", "arbitrary_tensor", "token_type_ids",
                     "num_soft_tokens_per_image", "num_soft_tokens_per_video", "personality_state"):
            with self.subTest(name=name), self.assertRaisesRegex(backbone.BackboneError, "known Gemma 4"):
                self.n0(input_ids=self.ids, attention_mask=self.mask, **{name: media})
        self.assertIsNone(self.representation.inputs)

    def test_rejects_invalid_or_hidden_source_rows_before_forward(self):
        invalid = [
            (torch.empty((0, 3), dtype=torch.int64), torch.empty((0, 3))),
            (torch.empty((1, 0), dtype=torch.int64), torch.empty((1, 0))),
            (self.ids.flatten(), self.mask.flatten()),
            (self.ids, self.mask[:, :2]),
            (self.ids, torch.zeros_like(self.mask)),
            (self.ids, self.mask.float() + 0.5),
            (self.ids, torch.full_like(self.mask, float("nan"), dtype=torch.float32)),
            (self.ids.float(), self.mask),
            (-self.ids, self.mask),
        ]
        for ids, mask in invalid:
            with self.subTest(ids_shape=ids.shape, mask_shape=mask.shape), \
                    self.assertRaises(backbone.BackboneError):
                self.n0(input_ids=ids, attention_mask=mask)
        self.assertIsNone(self.representation.inputs)

    def test_context_overflow_fails_without_source_truncation(self):
        ids = torch.ones((1, 6), dtype=torch.int64)
        with self.assertRaisesRegex(backbone.BackboneError, "truncation"):
            self.n0(input_ids=ids, attention_mask=torch.ones_like(ids))
        self.assertIsNone(self.representation.inputs)

    def test_no_decoder_or_string_configuration_passes_feature_boundary(self):
        for extra in ({"labels": self.ids}, {"inputs_embeds": torch.zeros(2, 3, 4)},
                      {"use_cache": True}, {"chat_template": "pretend to be Alice"},
                      {"generation_config": self.ids}):
            with self.subTest(extra=tuple(extra)), self.assertRaises(backbone.BackboneError):
                self.n0(input_ids=self.ids, attention_mask=self.mask, **extra)
        self.assertIsNone(self.representation.inputs)

    def test_hidden_shape_or_nonfinite_values_are_not_admitted(self):
        for output in (torch.zeros(2, 2, 4), torch.zeros(2, 3, 5),
                       torch.zeros(2, 3, 4, dtype=torch.int64),
                       torch.full((2, 3, 4), float("nan")),
                       torch.full((2, 3, 4), float("inf"))):
            self.representation.output_override = output
            with self.subTest(shape=output.shape), self.assertRaises(backbone.BackboneError):
                self.n0(input_ids=self.ids, attention_mask=self.mask)


@unittest.skipIf(torch is None, "install CPU Torch to verify prepared local loader")
class PreparedLoaderTests(unittest.TestCase):
    def setUp(self):
        class Representation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)

            def forward(self, **inputs):
                return types.SimpleNamespace(last_hidden_state=self.embedding(inputs["input_ids"]))

        class FullModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Representation()
                self.lm_head = torch.nn.Linear(4, 16)
                self.config = types.SimpleNamespace(
                    model_type="gemma4_unified", text_config=types.SimpleNamespace(
                        hidden_size=4, max_position_embeddings=8))

            def forward(self, *args, **kwargs):
                raise AssertionError("publisher LM forward was used")

            def generate(self, *args, **kwargs):
                raise AssertionError("publisher generation was used")

        self.full = FullModel()
        self.calls = []
        self.transformers = types.SimpleNamespace(__version__="fixture-version",
            AutoModelForMultimodalLM=types.SimpleNamespace(from_pretrained=self.load))
        self.receipt = {"snapshot_path": "/public/tiny-test-fixture", "runtime": {
            "backend": "transformers", "dtype": "float32",
            "torch_version": str(torch.__version__), "transformers_version": "fixture-version"}}

    def load(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self.full

    def imported(self, name):
        return {"torch": torch, "transformers": self.transformers}[name]

    def test_verified_loader_uses_safe_local_flags_and_drops_lm_object(self):
        with patch.object(preparation, "verify_prepared", return_value=self.receipt) as verify, \
                patch.object(backbone, "import_module", side_effect=self.imported):
            n0 = backbone.load_prepared_gemma_n0("prepared.json")
        self.assertEqual(verify.call_args_list, [
            unittest.mock.call("prepared.json"),
            unittest.mock.call("prepared.json", expected_runtime=self.receipt["runtime"]),
        ])
        self.assertEqual(self.calls, [(self.receipt["snapshot_path"], {
            "local_files_only": True, "trust_remote_code": False,
            "use_safetensors": True, "dtype": torch.float32})])
        self.assertFalse(hasattr(n0, "lm_head"))
        self.assertFalse(hasattr(n0, "generate"))
        self.assertFalse(hasattr(n0, "base"))
        self.assertFalse(hasattr(n0._representation, "lm_head"))
        self.assertIs(n0._representation, self.full.model)
        self.assertTrue(all(not p.requires_grad for p in self.full.parameters()))
        states = n0(input_ids=torch.tensor([[1, 2]]),
                    attention_mask=torch.ones(1, 2)).hidden_states
        self.assertEqual(states.shape, (1, 2, 4))

    def test_receipt_identity_change_during_load_cannot_return_a_wrapper(self):
        changed = {**self.receipt, "receipt_sha256": "changed-during-load"}
        with patch.object(preparation, "verify_prepared", side_effect=[self.receipt, changed]), \
                patch.object(backbone, "import_module", side_effect=self.imported):
            with self.assertRaisesRegex(backbone.BackboneError, "artifact changed while loading"):
                backbone.load_prepared_gemma_n0("prepared.json")
        self.assertEqual(len(self.calls), 1)

    def test_persistent_snapshot_code_or_receipt_mutation_during_load_fails_readmission(self):
        # Tiny public custody files and fixture code exercise the real verifier;
        # the model loader itself is still a small CPU fake, not 12B inference.
        for target in ("snapshot", "code", "receipt"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "source"
                source.mkdir()
                config = {"architectures": ["Gemma4UnifiedForConditionalGeneration"],
                          "model_type": "gemma4_unified"}
                contents = {"config.json": json.dumps(config).encode(),
                            "model.safetensors": b"public fixture tensor bytes",
                            "README.md": b"public model notice", ".gitattributes": b"public lfs notice"}
                for name, payload in contents.items():
                    (source / name).write_bytes(payload)
                pins = {name: (len(payload), sha256(payload).hexdigest())
                        for name, payload in contents.items()}
                code_paths = {}
                for name in ("preparation.py", "backbone.py"):
                    path = root / name
                    path.write_text("# public code fixture\n", encoding="utf-8")
                    code_paths[name] = path
                clone = root / "clone"
                clone_receipt = root / "clone.json"
                prepared_receipt = root / "prepared.json"
                with patch.dict(foundation.PINNED_FILES, pins, clear=True), \
                        patch.dict(preparation.IMPLEMENTATION_PATHS, code_paths, clear=True):
                    foundation.clone_verified_source(source, clone, clone_receipt, role="personality")
                    preparation.prepare(clone_receipt, prepared_receipt, runtime=self.receipt["runtime"])

                    def mutate_and_load(path, **kwargs):
                        self.calls.append((path, kwargs))
                        if target == "snapshot":
                            (clone / "model.safetensors").write_bytes(b"persistently changed weights")
                        elif target == "code":
                            code_paths["backbone.py"].write_text("# persistently changed code\n", encoding="utf-8")
                        else:
                            prepared_receipt.write_text('{"changed":"receipt"}', encoding="utf-8")
                        return self.full

                    with patch.object(self.transformers.AutoModelForMultimodalLM,
                                      "from_pretrained", side_effect=mutate_and_load), \
                            patch.object(backbone, "import_module", side_effect=self.imported):
                        with self.assertRaises(preparation.PreparationError):
                            backbone.load_prepared_gemma_n0(prepared_receipt)

    def test_runtime_mismatch_stops_before_loading_weight_files(self):
        self.receipt["runtime"]["torch_version"] = "unmatched-version"
        with patch.object(preparation, "verify_prepared", return_value=self.receipt), \
                patch.object(backbone, "import_module", side_effect=self.imported):
            with self.assertRaisesRegex(backbone.BackboneError, "installed runtime"):
                backbone.load_prepared_gemma_n0("prepared.json")
        self.assertFalse(self.calls)

    def test_wrong_model_architecture_is_rejected(self):
        self.full.config.model_type = "different_model"
        with patch.object(preparation, "verify_prepared", return_value=self.receipt), \
                patch.object(backbone, "import_module", side_effect=self.imported):
            with self.assertRaisesRegex(backbone.BackboneError, "representation path"):
                backbone.load_prepared_gemma_n0("prepared.json")

    def test_missing_explicit_context_budget_is_rejected(self):
        self.full.config.text_config.max_position_embeddings = None
        with patch.object(preparation, "verify_prepared", return_value=self.receipt), \
                patch.object(backbone, "import_module", side_effect=self.imported):
            with self.assertRaisesRegex(backbone.BackboneError, "dimensions"):
                backbone.load_prepared_gemma_n0("prepared.json")


if __name__ == "__main__":
    unittest.main()
