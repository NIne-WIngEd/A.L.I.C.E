from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.mfm.evaluate_base_behavior import _input_bytes, _rows, compare, load_diagnostic
from scripts.mfm.run_gemma4_base_behavior import BaselineError
from scripts.mfm import run_gemma4_prepared_base_behavior as prepared


class _Input:
    shape = (1, 3)

    def to(self, device):
        return self


class _Tokens:
    def __getitem__(self, index):
        return self

    def tolist(self):
        return [101, 102]


class _Processor:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(
            decode=lambda tokens, **kwargs: '{"proposals":[],"dispositions":[]}')

    def __call__(self, *, text, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": _Input()}


class _Model:
    def parameters(self):
        yield types.SimpleNamespace(device="cuda:0")

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        return _Tokens()


class _Torch:
    cuda = types.SimpleNamespace(manual_seed_all=lambda seed: None)
    manual_seed = staticmethod(lambda seed: None)
    inference_mode = staticmethod(nullcontext)


class PreparedBaseDiagnosticTests(unittest.TestCase):
    def test_pairs_exact_public_inputs_without_claiming_base_qualification(self):
        cases = load_diagnostic()
        emitted = _input_bytes(cases)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "prepared"
            snapshot.mkdir()
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            prompts = root / "public.jsonl"
            prompts.write_bytes(emitted)
            output = root / "result.jsonl"
            derivative = {"schema": "alice-gemma4-v1-clone-v1",
                          "receipt_sha256": "c" * 64,
                          "parent_source_receipt_sha256": "a" * 64}
            with patch.object(prepared, "_configure_offline_process",
                              return_value="network_interfaces_present_public_only"), \
                    patch.object(prepared, "_prepared_receipt", return_value=(derivative, "d" * 64)), \
                    patch.object(prepared, "_load_local_model", return_value=(_Torch, _Processor(), _Model())):
                result = prepared.run(snapshot, receipt, prompts, output,
                                      max_new_tokens=42, seed=17)
            self.assertFalse(result["qualification_claim"])
            self.assertEqual(result["cases"], 7)
            rows = _rows(output, cases)
            expected_prompt_digest = sha256(emitted).hexdigest()
            self.assertEqual({r["prompt_set_sha256"] for r in rows.values()},
                             {expected_prompt_digest})
            self.assertEqual({r["model_artifact_digest"] for r in rows.values()},
                             {"d" * 64})
            self.assertEqual({r["base_source_sha256"] for r in rows.values()},
                             {prepared.FILES["model.safetensors"][1]})
            self.assertEqual({r["network_boundary"] for r in rows.values()},
                             {"network_interfaces_present_public_only"})
            self.assertEqual(rows["blank-image-no-biography"]["status"], "unexercised")
            untouched = {case_id: {
                **row, "model_artifact_digest": prepared.FILES["model.safetensors"][1],
                "model_receipt": "a" * 64,
            } for case_id, row in rows.items()}
            report = compare(cases, untouched, rows)
            self.assertTrue(report["paired_control_verified"])
            self.assertFalse(report["qualification_claim"])

    def test_mfm_role_clone_and_modified_derivative_admit_with_pinned_ancestry(self):
        pinned = prepared.FILES["model.safetensors"][1]
        receipt = {"schema": "alice-gemma4-v1-clone-v1", "role": "mfm",
                   "repository": prepared.REPO, "revision": prepared.REVISION,
                   "parent_source_receipt_sha256": "a" * 64,
                   "files": [{"path": "model.safetensors", "sha256": pinned}]}
        module = types.ModuleType("alice_foundation.gemma4_v1")
        def verified(*args, **kwargs):
            self.assertEqual(kwargs["expected_role"], "mfm")
            return receipt
        module.verify_role_base = verified
        package = types.ModuleType("alice_foundation")
        package.__path__ = []
        with patch.dict("sys.modules", {"alice_foundation": package,
                                        "alice_foundation.gemma4_v1": module}):
            _, digest = prepared._prepared_receipt(Path("/fake"), Path("/fake/receipt"))
            self.assertEqual(digest, pinned)
            receipt["schema"] = "alice-gemma4-v1-source-v1"
            with self.assertRaisesRegex(BaselineError, "not a verified MFM role copy"):
                prepared._prepared_receipt(Path("/fake"), Path("/fake/receipt"))
            receipt["schema"] = "alice-gemma4-v1-clone-v1"
            receipt["role"] = "personality"
            with self.assertRaisesRegex(BaselineError, "lineage or status"):
                prepared._prepared_receipt(Path("/fake"), Path("/fake/receipt"))
            receipt["role"] = "mfm"
            receipt["schema"] = "alice-gemma4-v1-derivative-v1"
            receipt["files"][0]["sha256"] = "d" * 64
            receipt["upstream_weight_sha256"] = pinned
            receipt["qualification"] = "unqualified"
            _, digest = prepared._prepared_receipt(Path("/fake"), Path("/fake/receipt"))
            self.assertEqual(digest, "d" * 64)
            receipt["files"][0]["sha256"] = pinned
            with self.assertRaisesRegex(BaselineError, "changed pinned ancestry"):
                prepared._prepared_receipt(Path("/fake"), Path("/fake/receipt"))
            receipt["files"][0]["sha256"] = "d" * 64
            receipt["qualification"] = "qualified"
            with self.assertRaisesRegex(BaselineError, "changed pinned ancestry"):
                prepared._prepared_receipt(Path("/fake"), Path("/fake/receipt"))

    def test_public_route_records_network_and_strict_flag_rejects_active_interface(self):
        with patch.object(prepared.socket, "if_nameindex", return_value=[(1, "lo"), (2, "eth0")]):
            self.assertEqual(prepared._configure_offline_process(
                require_network_isolation=False), "network_interfaces_present_public_only")
            with self.assertRaisesRegex(BaselineError, "loopback-only"):
                prepared._configure_offline_process(require_network_isolation=True)
        with patch.object(prepared.socket, "if_nameindex", return_value=[(1, "lo")]):
            self.assertEqual(prepared._configure_offline_process(
                require_network_isolation=True),
                "loopback_only_interface_check_unix_proxy_unverified")

    def test_private_prompt_classification_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "prepared"
            snapshot.mkdir()
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            prompts = root / "private.jsonl"
            prompts.write_text(json.dumps({"case_id": "private-case",
                                           "classification": "private",
                                           "prompt": "do not open"}) + "\n")
            with patch.object(prepared, "_configure_offline_process",
                              return_value="network_interfaces_present_public_only"), \
                    patch.object(prepared, "_prepared_receipt", return_value=(
                        {"receipt_sha256": "c" * 64}, "d" * 64)), \
                    patch.object(prepared, "_load_local_model",
                                 side_effect=AssertionError("model must not load")):
                with self.assertRaisesRegex(BaselineError, "public_synthetic"):
                    prepared.run(snapshot, receipt, prompts, root / "out.jsonl",
                                 max_new_tokens=4, seed=0)

    def test_public_label_cannot_admit_a_changed_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "prepared"
            snapshot.mkdir()
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            prompts = root / "mislabeled.jsonl"
            prompts.write_text(json.dumps({"case_id": "other-case",
                                           "classification": "public_synthetic",
                                           "prompt": "different content"}) + "\n")
            with patch.object(prepared, "_configure_offline_process",
                              return_value="network_interfaces_present_public_only"), \
                    patch.object(prepared, "_prepared_receipt", return_value=(
                        {"receipt_sha256": "c" * 64}, "d" * 64)), \
                    patch.object(prepared, "_load_local_model",
                                 side_effect=AssertionError("model must not load")):
                with self.assertRaisesRegex(BaselineError, "frozen public synthetic"):
                    prepared.run(snapshot, receipt, prompts, root / "out.jsonl",
                                 max_new_tokens=4, seed=0)


if __name__ == "__main__":
    unittest.main()
