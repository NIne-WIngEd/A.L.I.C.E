"""Actual fictional optimization -> governed words -> reviewed outcome mechanics."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from src.alice_personality.identity import IdentityError, IdentityModel, IdentityModelConfig
from src.alice_personality.identity.consumer import CheckpointPin, DecisionConsumer, RenderedResponse
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.teaching.contracts import TeachingBatch, TrainingRecipe
from src.alice_personality.teaching.trainer import IdentityTrainer
from tests.personality.test_identity_codec import Backbone, Tokenizer, document
from tests.personality.test_identity_teaching import _admission, _target


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(411)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.config = IdentityModelConfig(8, 3, learned_width=16, attention_heads=4)
        self.doc = document()
        self.doc = replace(self.doc, candidates=(
            replace(self.doc.candidates[0], text="I will agree without checking."),
            replace(self.doc.candidates[1], text="I disagree. Let's check the evidence; the choice remains yours.",
                    authority_allowed=True)))
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.config, _fixture=True)
        self.produced = producer.produce(self.doc)
        self.session = producer.close_and_verify()
        self.model = IdentityModel(self.config)

    def build(self, *, tie=False, co_valid=False):
        if tie:
            with torch.no_grad():
                self.model.n2.scalar[-1].weight[0].zero_()
                self.model.n2.scalar[-1].bias[0].zero_()
        target = [[.5, .5]] if tie else [[.02, .98]]
        targets = {"preferences": _target(target)}
        if co_valid:
            targets['co_valid_probabilities'] = _target([[1., 1.]])
        batch = TeachingBatch(self.produced.frame, targets,
            ("fictional-judgment",), (("fictional-source-family",),), "n2", "TRAIN", "public_mechanical_fixture")
        recipe = TrainingRecipe("n2", seed=411, learning_rate=.01, max_grad_norm=1, weight_decay=0)
        admission = _admission(self.path, self.model, recipe, [batch])
        trainer = IdentityTrainer(self.model, recipe, admission, first_batch=batch)
        for _ in range(60 if not tie else 1):
            trainer.step(batch)
        manifest = trainer.save_checkpoint(self.path / "checkpoint")
        return DecisionConsumer(self.model, (CheckpointPin(str(self.path / "checkpoint"), manifest["manifest_sha256"]),))

    def test_real_optimized_decision_selects_words_and_preserves_disagreement(self):
        consumer = self.build()
        result = consumer.respond(self.doc, self.produced, self.session)
        self.assertEqual(result["selected_candidate_ids"], ["opaque-candidate-b"])
        self.assertEqual(result["response_text"], self.doc.candidates[1].text)
        self.assertIn("choice remains yours", result["response_text"])
        self.assertEqual(result["intents"][1]["voice_controls"], {})
        self.assertEqual(result["training_state"], "OPTIMIZED_UNQUALIFIED")
        self.assertFalse(result["acceptance_authority"])
        self.assertEqual(result["rendering"]["status"], "NOT_RENDERED")

    def test_equal_preferences_are_unresolved_without_candidate_order_tiebreak(self):
        result = self.build(tie=True).respond(self.doc, self.produced, self.session)
        self.assertEqual(result["status"], "UNRESOLVED_PLURALITY")
        self.assertEqual(set(result["selected_candidate_ids"]), {e.candidate_id for e in self.doc.candidates})
        self.assertIsNone(result["response_text"])

    def test_valid_alternative_does_not_erase_learned_preference(self):
        result = self.build(co_valid=True).respond(self.doc, self.produced, self.session)
        self.assertEqual(result['status'], 'SELECTED')
        self.assertEqual(result['selected_candidate_ids'], ['opaque-candidate-b'])
        self.assertEqual(set(result['acceptable_candidate_ids']), {'opaque-candidate-a', 'opaque-candidate-b'})

    def test_changed_renderer_words_are_retained_as_unresolved(self):
        result = self.build().respond(self.doc, self.produced, self.session,
            renderer=lambda intent: RenderedResponse("Certainly, I agree.", {}, "fictional-renderer", None))
        self.assertEqual(result["rendering"]["status"], "UNRESOLVED_RENDERER")
        self.assertEqual(result["rendering"]["text"], "Certainly, I agree.")
        self.assertEqual(result["response_text"], self.doc.candidates[1].text)

    def test_changed_model_document_features_or_checkpoint_cannot_reuse_evidence(self):
        consumer = self.build()
        changed = replace(self.doc, candidates=(replace(self.doc.candidates[0], text="changed"), self.doc.candidates[1]))
        with self.assertRaisesRegex(IdentityError, "document"):
            consumer.respond(changed, self.produced, self.session)
        with torch.no_grad():
            self.model.n2.scalar[-1].bias[0].add_(1)
        with self.assertRaisesRegex(IdentityError, "model state"):
            consumer.respond(self.doc, self.produced, self.session)

    def test_unoptimized_checkpoint_cannot_present_learned_response(self):
        batch = TeachingBatch(self.produced.frame, {"preferences": _target([[.02, .98]])},
            ("fictional-judgment",), (("fictional-source-family",),), "n2", "TRAIN", "public_mechanical_fixture")
        recipe = TrainingRecipe("n2", learning_rate=.01, max_grad_norm=1, seed=411)
        trainer = IdentityTrainer(self.model, recipe, _admission(self.path, self.model, recipe, [batch]), first_batch=batch)
        manifest = trainer.save_checkpoint(self.path / "empty")
        with self.assertRaisesRegex(IdentityError, "optimized"):
            DecisionConsumer(self.model, (CheckpointPin(str(self.path / "empty"), manifest["manifest_sha256"]),))
