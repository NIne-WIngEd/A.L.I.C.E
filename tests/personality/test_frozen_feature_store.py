"""One immutable complete feature per exact input under caller custody."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from src.alice_personality.identity import IdentityError, IdentityModelConfig
from src.alice_personality.identity.feature_store import FrozenFeatureStore
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from tests.personality.test_identity_codec import Backbone, Tokenizer, document


class FeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.config = IdentityModelConfig(8, 3, learned_width=16, attention_heads=4)
        self.backbone = Backbone()
        self.store = FrozenFeatureStore.create(self.path / "store", source_package_sha256="a" * 64)
        self.producer = FrozenFrameProducer(Tokenizer(), self.backbone, self.config, _fixture=True, feature_store=self.store)

    def test_shared_entries_extract_once_and_replay_without_publisher_or_duplicate_frames(self):
        first = self.producer.produce(document())
        count = self.backbone.calls
        changed = replace(document(), fields=tuple(replace(f, text="A new complete situation.")
            if f.semantic_role == "situation" else f for f in document().fields))
        second = self.producer.produce(changed)
        self.assertEqual(self.backbone.calls, count + 1)
        self.assertTrue(torch.equal(first.frame.sources.states, second.frame.sources.states))
        session = self.producer.close_and_verify()
        self.assertTrue(session["persistent_feature_values_retained"])
        self.assertEqual(session["feature_store"]["unique_entries"], count + 1)
        loaded = FrozenFeatureStore.open(self.path / "store", expected_manifest_sha256=session["feature_store"]["manifest_sha256"])
        replay = FrozenFrameProducer.replay(loaded, self.config, session, expected_session_sha256=session["receipt_sha256"])
        restored = replay.produce(document())
        self.assertEqual(restored.binding, first.binding)
        torch.testing.assert_close(restored.frame.sources.states, first.frame.sources.states, atol=0, rtol=0)
        self.assertEqual(self.backbone.calls, count + 1)

    def test_changed_inputs_masks_context_or_payload_cannot_be_reused(self):
        self.producer.produce(document())
        session = self.producer.close_and_verify()
        loaded = FrozenFeatureStore.open(self.path / "store", expected_manifest_sha256=session["feature_store"]["manifest_sha256"])
        replay = FrozenFrameProducer.replay(loaded, self.config, session, expected_session_sha256=session["receipt_sha256"])
        changed = replace(document(), candidates=(replace(document().candidates[0], text="never extracted"), document().candidates[1]))
        with self.assertRaisesRegex(IdentityError, "complete|entry"):
            replay.produce(changed)
        with self.assertRaisesRegex(IdentityError, "context"):
            loaded.bind_context({"encoder": "changed"})
        entry = next((self.path / "store").glob("*.pt"))
        with entry.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(IdentityError, "payload"):
            replay.produce(document())

    def test_every_actual_layer_and_token_remain_in_store(self):
        first = self.producer.produce(document())
        session = self.producer.close_and_verify()
        payload = torch.load(next((self.path / "store").glob("*.pt")), weights_only=True)
        self.assertEqual(payload["states"].shape[:2], (1, 3))
        self.assertEqual(payload["states"].shape[2], payload["input_ids"].shape[1])
        self.assertEqual(payload["states"].shape[-1], 8)
        self.assertEqual(session["feature_store"]["source_package_sha256"], "a" * 64)
        self.assertFalse(first.frame.sources.states.requires_grad)

    def test_same_text_with_changed_token_ids_or_mask_refuses_stale_values(self):
        self.producer.produce(document())
        text = next(iter(self.producer.tokenizer.seen))
        inputs = self.producer._tokenize(text)
        changed = {k: v.clone() for k, v in inputs.items()}
        changed['input_ids'][0, 0] += 1
        with self.assertRaisesRegex(IdentityError, 'token IDs or attention mask'):
            self.store.features(text, changed, lambda: self.fail('must not silently recompute'))
        changed = {k: v.clone() for k, v in inputs.items()}
        changed['attention_mask'][0, 0] = 0
        with self.assertRaisesRegex(IdentityError, 'token IDs or attention mask'):
            self.store.features(text, changed, lambda: self.fail('must not silently recompute'))
