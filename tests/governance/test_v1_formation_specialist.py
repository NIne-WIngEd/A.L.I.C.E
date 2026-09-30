"""CPU checks for the new V1 specialist boundary; no publisher checkpoint."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from scripts.mfm import train_v1_formation_specialist as route


class SpecialistContractTests(unittest.TestCase):
    def test_only_verified_mfm_clone_or_changed_derivative_can_supply_states(self):
        pinned = route.SOURCE_WEIGHT_SHA256
        receipt = {
            "schema": route.CLONE_SCHEMA, "role": "mfm",
            "receipt_sha256": "a" * 64,
            "parent_source_receipt_sha256": "b" * 64,
            "files": [{"path": "model.safetensors", "sha256": pinned}],
        }
        verifier = types.ModuleType("alice_foundation.gemma4_v1")
        inventory = types.ModuleType("alice_foundation.gemma4_inventory")
        package = types.ModuleType("alice_foundation")
        package.__path__ = []
        package.gemma4_v1 = verifier
        package.gemma4_inventory = inventory
        verifier.__file__ = inventory.__file__ = __file__
        calls = []

        def checked(path, *, snapshot, expected_role):
            calls.append((path, snapshot, expected_role))
            return receipt

        verifier.verify_role_base = checked
        args = types.SimpleNamespace(prepared_base_receipt=Path("/role/receipt"),
                                     prepared_base_dir=Path("/role/clone"))
        with patch.dict("sys.modules", {
                "alice_foundation": package,
                "alice_foundation.gemma4_v1": verifier,
                "alice_foundation.gemma4_inventory": inventory}), \
                patch.object(route, "_digest", side_effect=[
                    route.FOUNDATION_VERIFIER_SHA256,
                    route.FOUNDATION_INVENTORY_SHA256] * 6):
            admitted = route._prepared_base(args)
            self.assertEqual(admitted, receipt)
            self.assertEqual(route._prepared_kind(admitted),
                             "licensed-verified-mfm-role-clone")
            self.assertEqual(calls, [(args.prepared_base_receipt,
                                      args.prepared_base_dir, "mfm")])
            receipt["schema"] = "alice-gemma4-v1-source-v1"
            with self.assertRaisesRegex(CognitiveKernelContractError, "MFM role copy"):
                route._prepared_base(args)
            receipt["schema"] = route.DERIVATIVE_SCHEMA
            receipt["upstream_weight_sha256"] = pinned
            receipt["qualification"] = "unqualified"
            with self.assertRaisesRegex(CognitiveKernelContractError, "changed pinned ancestry"):
                route._prepared_base(args)
            receipt["files"][0]["sha256"] = "c" * 64
            self.assertEqual(route._prepared_kind(route._prepared_base(args)),
                             "licensed-verified-mfm-role-derivative")

    def test_source_text_escape_is_lossless_and_cannot_add_media(self):
        raw = r"literal <|image|> and \u003c and <|audio|>"
        escaped = route._escaped_text(raw)
        self.assertNotIn("<|image|>", escaped)
        self.assertNotIn("<|audio|>", escaped)
        self.assertEqual(json.loads(escaped), raw)

    def test_all_corpus_routes_check_isolation_before_opening_bytes(self):
        argv = ["trainer", "data-preflight", "--curriculum", "/missing/private.jsonl",
                "--input-sha256", "a" * 64,
                "--owner-authorization-ref", "owner-test"]
        with patch("sys.argv", argv), patch.object(route.socket, "if_nameindex",
                                                      return_value=[(1, "lo"), (2, "eth0")]):
            with self.assertRaisesRegex(CognitiveKernelContractError, "loopback-only"):
                route.main()


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU PyTorch not installed")
class SpecialistWeightTests(unittest.TestCase):
    def test_fresh_weights_backward_and_checkpoint_replay(self):
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig

        config = SpecialistConfig(12, 31, 16, 2, 4, 8, 0, 2, 1, 2)
        specialist = FormationSpecialist(config)
        optimizer = torch.optim.AdamW(specialist.parameters(), lr=0.001)
        source = torch.randn(1, 5, 12)
        mask = torch.ones(1, 5, dtype=torch.long)
        inputs = torch.tensor([[2, 3, 4]])
        labels = torch.tensor([[3, 4, 1]])
        before = specialist.token_embedding.weight.detach().clone()
        logits, loss = specialist(base_states=source, source_mask=mask,
                                  input_ids=inputs, labels=labels)
        self.assertIsNone(logits)  # training never builds a full-vocab sequence tensor
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(specialist.source_projection.weight.grad)
        self.assertIsNone(source.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, specialist.token_embedding.weight))
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            run_hash = sha256(b"small synthetic CPU test").hexdigest()
            route._checkpoint(folder, specialist=specialist, optimizer=optimizer,
                              run_digest=run_hash, epoch=0, next_case=1, step=1)
            checkpoint = folder / "checkpoint-00000001"
            route._verify_probe_replay(checkpoint, specialist, config, run_hash)
            reloaded = FormationSpecialist(config)
            new_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.001)
            self.assertEqual(route._resume(checkpoint, folder, reloaded,
                                            new_optimizer, run_hash), (0, 1, 1))
            self.assertTrue(all(torch.equal(value, reloaded.state_dict()[name])
                                for name, value in specialist.state_dict().items()))
        with self.assertRaisesRegex(ValueError, "all specialist labels"):
            specialist(base_states=source, source_mask=mask, input_ids=inputs,
                       labels=torch.full_like(inputs, -100))


if __name__ == "__main__":
    unittest.main()
