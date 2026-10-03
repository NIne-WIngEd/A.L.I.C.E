"""Public fictional packaging tests; no Gemma/identity qualification."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest

import torch

from src.alice_personality.identity.codec import (REQUIRED_ROLES, CandidateEntry, ConceptEntry,
    CognitiveFrameDocument, GraphEdge, SemanticEntry, SituationField, SourceEntry, decision_to_mapping, fingerprint)
from src.alice_personality.identity.contracts import (HEAD_FAMILIES, ConceptRecord,
    IdentityError, IdentityModelConfig, LabelSpec, SourceRecord)
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.identity.model import IdentityModel


def document():
    fields = tuple(SituationField(role + "-id", role, role.replace("_", " "),
                   "A fictional person considers sharing an unrelated rumor." if role == "situation" else None,
                   role == "situation") for role in sorted(REQUIRED_ROLES))
    source = SourceRecord("opaque-source-id", "fictional-source-family", "E0", "E0",
                          True, False, True, False, False, False)
    labels = {family: (LabelSpec(family + "-id", family, "A supplied semantic " + family + " descriptor",
        "normalized_intensity" if family == "voice" else "categorical",
        0 if family == "voice" else None, 1 if family == "voice" else None),) for family in HEAD_FAMILIES}
    return CognitiveFrameDocument(fields, (SourceEntry("Fictional source favors evidence before disclosure.", source),),
        (ConceptEntry("Privacy in unverified disclosure.", ConceptRecord("opaque-concept-id", "identity", True)),),
        (CandidateEntry("opaque-candidate-a", "Ask for reliable evidence before disclosure.", True, True),
         CandidateEntry("opaque-candidate-b", "Disclose the rumor immediately.", False, True)),
        (SemanticEntry("opaque-relation-id", "Supports the conditional source policy."),), labels,
        (GraphEdge("opaque-source-id", "opaque-concept-id", "opaque-relation-id", "identity_support"),))


class Tokenizer:
    def __init__(self):
        self.seen = []
    def __call__(self, texts, **kwargs):
        assert kwargs == dict(padding=False, truncation=False, add_special_tokens=True,
                              return_attention_mask=True, return_tensors="pt")
        text, = texts
        self.seen.append(text)
        size = 65 if "over-budget" in text else 4
        ids = torch.tensor([[2] + [3 + b for b in sha256(text.encode()).digest()][:size - 2] + [1]])
        if size == 65:
            ids = torch.ones(1, size, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class Backbone:
    hidden_size, hidden_state_count, max_source_tokens, device = 8, 3, 64, torch.device("cpu")
    def __init__(self):
        self.calls = 0
    def extract_features(self, input_ids, attention_mask):
        self.calls += 1
        values = input_ids.float()[..., None].expand(-1, -1, 8) / 100
        states = tuple((values + index).bfloat16().requires_grad_(True) for index in range(3))
        return SimpleNamespace(hidden_states=states[-1], all_hidden_states=states,
                               attention_mask=attention_mask.bool())


class CodecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)
        self.document = document()
        self.config = IdentityModelConfig(8, 3, learned_width=16, attention_heads=4)
        self.tokenizer, self.backbone = Tokenizer(), Backbone()
        self.producer = FrozenFrameProducer(self.tokenizer, self.backbone, self.config, _fixture=True)

    def test_full_document_json_roundtrip_and_target_separation(self):
        raw = json.loads(json.dumps(self.document.to_mapping()))
        restored = CognitiveFrameDocument.from_mapping(raw)
        self.assertEqual(restored.sha256, self.document.sha256)
        self.assertEqual(CognitiveFrameDocument.from_json(json.dumps(raw)).sha256, self.document.sha256)
        raw["preferred_candidate"] = "opaque-candidate-a"
        with self.assertRaisesRegex(IdentityError, "target-free"):
            CognitiveFrameDocument.from_mapping(raw)
        with self.assertRaisesRegex(IdentityError, "duplicate"):
            CognitiveFrameDocument.from_json('{"schema":"x","schema":"y"}')

    def test_explicit_unknown_fields_and_strict_authority_flags(self):
        with self.assertRaisesRegex(IdentityError, "full frame roles"):
            replace(self.document, fields=self.document.fields[:1]).validate()
        field = replace(self.document.fields[0], available=False, text="hidden context")
        with self.assertRaisesRegex(IdentityError, "hidden content"):
            replace(self.document, fields=(field, *self.document.fields[1:])).validate()
        entry = replace(self.document.sources[0], record=replace(self.document.sources[0].record,
                                                                 identity_core_allowed="false"))
        with self.assertRaisesRegex(IdentityError, "actual boolean"):
            replace(self.document, sources=(entry,)).validate()

    def test_real_tensor_path_all_layers_masks_pointer_ids_and_full_output(self):
        produced = self.producer.produce(self.document)
        frame = produced.frame
        self.assertEqual(frame.sources.states.shape, (1, 1, 3, 4, 8))
        self.assertEqual(frame.sources.states.dtype, torch.bfloat16)
        self.assertFalse(frame.sources.states.requires_grad)
        self.assertEqual(produced.binding["frame_sha256"], fingerprint(frame))
        changed = replace(frame, candidate_authority_mask=~frame.candidate_authority_mask)
        self.assertNotEqual(produced.binding["frame_sha256"], fingerprint(changed))
        semantic = "\n".join(self.tokenizer.seen)
        self.assertNotIn("opaque-", semantic)
        self.assertIn("Evidence class: E0", semantic)
        self.assertIn("[not supplied]", semantic)
        packet = IdentityModel(self.config)(frame)
        self.assertEqual(packet.preferences[0, 1], 0)
        decoded = decision_to_mapping(packet)
        self.assertEqual(set(decoded["heads"]), set(HEAD_FAMILIES))
        self.assertEqual(decoded["candidate_ids"][0][0], "opaque-candidate-a")
        self.assertIn("voice_control_values", decoded)
        self.assertIn("historical_evidence_pointers", decoded)
        self.assertIn("native_latent", decoded)
        self.assertEqual(decoded["qualification"], "UNQUALIFIED")
        footer = self.producer.close_and_verify()
        self.assertEqual(footer["frame_binding_sha256"], [produced.binding["binding_sha256"]])
        self.assertFalse(footer["session_source_recheck_passed"])
        self.assertTrue(footer["mechanical_fixture_only"])
        with self.assertRaisesRegex(IdentityError, "closed"):
            self.producer.produce(self.document)

    def test_complete_overbudget_refusal_precedes_every_forward(self):
        entry = replace(self.document.candidates[0], text="over-budget")
        bad = replace(self.document, candidates=(entry, self.document.candidates[1]))
        with self.assertRaisesRegex(IdentityError, "truncation is forbidden"):
            self.producer.produce(bad)
        self.assertEqual(self.backbone.calls, 0)

    def test_missing_source_concept_relation_and_candidate_are_explicit_masks(self):
        empty = replace(self.document, sources=(), concepts=(), relations=(), edges=(), candidates=())
        frame = self.producer.produce(empty).frame
        packet = IdentityModel(self.config)(frame)
        self.assertFalse(packet.candidate_mask.any())
        self.assertFalse(packet.identity_grounding_available.any())
        self.assertTrue((packet.uncertainty == 1).all())
        self.assertEqual(frame.graph.senders.shape, (1, 0))

    def test_graph_cannot_promote_context_only_source_to_identity(self):
        source = replace(self.document.sources[0], record=replace(self.document.sources[0].record,
            source_kind="HOST", provenance_class="HOST", identity_core_allowed=False,
            context_allowed=True, historical_truth_allowed=False))
        with self.assertRaisesRegex(IdentityError, "identity graph support"):
            self.producer.produce(replace(self.document, sources=(source,)))
        self.assertEqual(self.backbone.calls, 0)

    def test_provider_missing_layer_or_changed_mask_is_refused(self):
        original = self.backbone.extract_features
        def incomplete(**inputs):
            value = original(**inputs)
            value.all_hidden_states = value.all_hidden_states[1:]
            return value
        self.backbone.extract_features = incomplete
        with self.assertRaisesRegex(IdentityError, "source geometry"):
            self.producer.produce(self.document)

    def test_code_mutation_binding_cannot_close_or_emit(self):
        self.producer._implementation["codec.py"] = "0" * 64
        with self.assertRaisesRegex(IdentityError, "implementation changed"):
            self.producer.produce(self.document)
        with self.assertRaisesRegex(IdentityError, "implementation changed"):
            self.producer.close_and_verify()

    def test_production_constructor_and_external_preparation_binding_fail_closed(self):
        with self.assertRaisesRegex(IdentityError, "verified loader"):
            FrozenFrameProducer(self.tokenizer, self.backbone, self.config)


if __name__ == "__main__":
    unittest.main()
