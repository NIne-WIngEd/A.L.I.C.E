"""Random public tensor mechanics, never identity learning or qualification."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.alice_personality.identity import (
    HEAD_FAMILIES, CalibrationBatch, ConceptRecord, FrozenTokenBank, IdentityError, IdentityFrame,
    IdentityModel, IdentityModelConfig, LabelSpec, SourceRecord, SupportGraph,
    load_untrained_snapshot, save_untrained_snapshot)


def _bank(batch, count, config, prefix, *, tokens=4, dtype=torch.float32):
    states = torch.randn(batch, count, config.provider_state_count, tokens, config.provider_width, dtype=dtype)
    token_mask = torch.ones(batch, count, tokens, dtype=torch.bool)
    return FrozenTokenBank(states, token_mask, token_mask.any(-1),
                           tuple(tuple(f"{prefix}-{index}" for index in range(count)) for _ in range(batch)))


def _frame(config, *, batch=2, candidates=3, dtype=torch.float32):
    query = _bank(batch, 2, config, "query", dtype=dtype)
    sources = _bank(batch, 5, config, "source", dtype=dtype)
    concepts = _bank(batch, 5, config, "concept", dtype=dtype)
    candidate_bank = _bank(batch, candidates, config, "candidate", dtype=dtype)
    relations = _bank(batch, 2, config, "relation", dtype=dtype)
    source_records = tuple(tuple([
        SourceRecord("source-0", "family-a", "E0", "E0", True, True, True, False, False, False),
        SourceRecord("source-1", "family-a", "EINF", "E-INF", True, True, False, False, False, False),
        SourceRecord("source-2", "family-host", "HOST", "HOST", False, True, False, False, False, False),
        SourceRecord("source-3", "family-relationship", "RELATIONSHIP", "RELATIONSHIP", False, True, False, False, False, False),
        SourceRecord("source-4", "family-self", "AEXP", "A-EXP", False, True, False, True, True, False),
    ]) for _ in range(batch))
    views = ("identity", "host", "relationship", "self", "context")
    concept_records = tuple(tuple(ConceptRecord(f"concept-{i}", view, residual=i == 0)
                                    for i, view in enumerate(views)) for _ in range(batch))
    labels, specs = {}, {}
    for family in HEAD_FAMILIES:
        labels[family] = _bank(batch, 3, config, family, dtype=dtype)
        specs[family] = tuple(tuple(LabelSpec(f"{family}-{i}", f"{family} named option {i}",
                                             f"Public fictional {family} semantic description {i}",
                                             "normalized_control" if family == "voice" else "categorical",
                                             0.0 if family == "voice" else None,
                                             1.0 if family == "voice" else None)
                                   for i in range(3)) for _ in range(batch))
    senders = torch.tensor([[0, 1, 2, 3, 4, 1]] * batch)
    receivers = torch.tensor([[5, 5, 6, 7, 8, 9]] * batch)
    relation_indices = torch.tensor([[0, 1, 0, 0, 1, 1]] * batch)
    roles = tuple(("identity_support", "inference_support", "context_only", "context_only", "context_only", "context_only")
                  for _ in range(batch))
    graph = SupportGraph(senders, receivers, relation_indices, torch.ones(batch, 6, dtype=torch.bool), roles)
    mask = torch.ones(batch, candidates, dtype=torch.bool)
    return IdentityFrame(query, sources, concepts, candidate_bank, relations, labels, source_records,
                         concept_records, specs, graph, mask.clone(), mask.clone())


def _permute_bank(bank, order):
    return replace(bank, states=bank.states[:, order], token_mask=bank.token_mask[:, order],
                   entry_mask=bank.entry_mask[:, order],
                   entry_ids=tuple(tuple(ids[i] for i in order) for ids in bank.entry_ids))


class IdentityModelMechanicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(20261002)
        self.config = IdentityModelConfig(6, 4, learned_width=16, attention_heads=4,
                                         graph_layers=2, readout_layers=2)
        self.frame = _frame(self.config)
        self.model = IdentityModel(self.config)

    def test_complete_multihead_dynamic_packet_is_explicitly_untrained(self):
        packet = self.model(self.frame)
        self.assertEqual(packet.training_state, "UNTRAINED")
        self.assertEqual(packet.qualification, "UNQUALIFIED")
        self.assertEqual(set(packet.heads), set(HEAD_FAMILIES))
        self.assertEqual(packet.preferences.shape, (2, 3))
        torch.testing.assert_close(packet.preferences.sum(-1), torch.ones(2))
        self.assertEqual(packet.co_valid_probabilities.shape, (2, 3))
        self.assertEqual(packet.value_tradeoff_margins.shape, (2, 3, 3, 3))
        self.assertEqual(packet.voice_control_values.shape, (2, 3, 3))
        self.assertEqual(packet.voice_control_confidence.shape, (2, 3, 3))
        self.assertEqual(packet.evidence_pointers.shape, (2, 3, 5))
        self.assertEqual(packet.concept_pointers.shape, (2, 3, 5))
        self.assertEqual(packet.latent_candidates.shape, (2, 3, 16))
        self.assertEqual(packet.representation.source_tokens.shape, (2, 5, 4, 16))
        self.assertEqual(packet.representation.layer_readout_weights.shape, (2, 5, 4, 4))
        self.assertEqual(packet.representation.concept_source_logits.shape, (2, 5, 5))
        self.assertFalse(packet.representation.concept_source_mask[:, 0, 2:].any())
        self.assertEqual(set(packet.representation.views), {"identity", "host", "relationship", "self", "context"})
        for family in HEAD_FAMILIES:
            self.assertEqual(packet.heads[family].scores.shape, (2, 3, 3))
            self.assertEqual(packet.heads[family].mask.dtype, torch.bool)
            self.assertTrue(torch.isfinite(packet.heads[family].scores).all())
        for name in ("evidence_sufficiency", "uncertainty", "contraindications", "failure_tail_risk", "owner_fidelity_uncertainty"):
            self.assertEqual(getattr(packet, name).shape, (2, 3))
        with self.assertRaises(ValueError):
            replace(packet, qualification="QUALIFIED")

    def test_first_party_gradients_do_not_reach_any_provider_bank(self):
        banks = (self.frame.query, self.frame.sources, self.frame.concepts, self.frame.candidates,
                 self.frame.relations, *self.frame.labels.values())
        for bank in banks:
            bank.states.requires_grad_(True)
        self.model.set_phase("n2", adapt_n1=True)
        packet = self.model(self.frame)
        objective = (packet.preference_logits.square().sum() + packet.latent_candidates.square().sum()
                     + sum(head.scores.square().sum() for head in packet.heads.values()))
        objective.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in self.model.n1.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in self.model.n2.parameters()))
        self.assertTrue(all(bank.states.grad is None for bank in banks))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.n3.parameters()))

    def test_bfloat16_provider_is_cast_to_first_party_float32_for_gradients(self):
        frame = _frame(self.config, dtype=torch.bfloat16)
        frame.sources.states.requires_grad_(True)
        self.model.set_phase("n2", adapt_n1=True)
        packet = self.model(frame)
        self.assertEqual(packet.latent_candidates.dtype, torch.float32)
        packet.heads["values"].scores.square().sum().backward()
        self.assertIsNone(frame.sources.states.grad)
        self.assertTrue(any(p.grad is not None for p in self.model.n1.parameters()))
        self.assertEqual(frame.sources.states.dtype, torch.bfloat16)

    def test_middle_provider_layer_has_real_mechanical_access(self):
        original = self.model(self.frame)
        states = self.frame.sources.states.clone()
        states[:, 0, 1] += torch.randn_like(states[:, 0, 1]) * 8
        changed = self.model(replace(self.frame, sources=replace(self.frame.sources, states=states)))
        self.assertFalse(torch.allclose(original.representation.views["identity"], changed.representation.views["identity"]))
        self.assertFalse(torch.allclose(original.preference_logits, changed.preference_logits))
        self.assertTrue((original.representation.layer_readout_weights[:, :, 1] > 0).all())

    def test_candidate_permutation_is_equivariant_across_packet_heads(self):
        original = self.model(self.frame)
        order = [2, 0, 1]
        frame = replace(self.frame, candidates=_permute_bank(self.frame.candidates, order),
                        candidate_authority_mask=self.frame.candidate_authority_mask[:, order],
                        candidate_context_mask=self.frame.candidate_context_mask[:, order])
        changed = self.model(frame)
        for name in ("preferences", "preference_logits", "co_valid_probabilities", "latent_candidates",
                     "uncertainty", "evidence_pointers", "concept_pointers", "voice_control_values"):
            torch.testing.assert_close(getattr(changed, name), getattr(original, name)[:, order], atol=1e-6, rtol=1e-5)
        for family in HEAD_FAMILIES:
            torch.testing.assert_close(changed.heads[family].scores, original.heads[family].scores[:, order], atol=1e-6, rtol=1e-5)
        self.assertEqual(changed.candidate_ids, frame.candidates.entry_ids)

    def test_authority_blocked_candidate_cannot_influence_allowed_choices(self):
        mask = self.frame.candidate_authority_mask.clone()
        mask[:, 1] = False
        frame = replace(self.frame, candidate_authority_mask=mask)
        first = self.model(frame)
        states = frame.candidates.states.clone()
        states[:, 1] += 100
        second = self.model(replace(frame, candidates=replace(frame.candidates, states=states)))
        torch.testing.assert_close(first.preferences, second.preferences)
        for family in HEAD_FAMILIES:
            self.assertTrue((first.heads[family].scores[:, 1] == 0).all())
            torch.testing.assert_close(first.heads[family].scores, second.heads[family].scores)
        self.assertTrue((first.preferences[:, 1] == 0).all())
        self.assertTrue((first.evidence_pointers[:, 1] == 0).all())
        self.assertTrue((first.uncertainty[:, 1] == 1).all())

    def test_host_and_self_views_do_not_rewrite_source_identity_or_history(self):
        original = self.model(self.frame)
        states = self.frame.sources.states.clone()
        states[:, 2:] += torch.randn_like(states[:, 2:]) * 10
        changed = self.model(replace(self.frame, sources=replace(self.frame.sources, states=states)))
        torch.testing.assert_close(original.representation.views["identity"], changed.representation.views["identity"])
        self.assertFalse(torch.allclose(original.representation.views["host"], changed.representation.views["host"]))
        self.assertTrue((changed.historical_evidence_pointers[..., 1:] == 0).all())
        self.assertTrue((changed.alice_experience_pointers[..., :4] == 0).all())
        for col, flag in ((2, "identity_core_allowed"), (2, "historical_truth_allowed"), (0, "alice_lived_memory")):
            rows = [list(row) for row in self.frame.source_records]
            rows[0][col] = replace(rows[0][col], **{flag: True})
            with self.assertRaises(IdentityError):
                self.model(replace(self.frame, source_records=tuple(tuple(row) for row in rows)))

    def test_synthetic_prior_never_promotes_itself_to_history(self):
        record = SourceRecord("synthetic", "family", "ASYN_TARGETED", "A-SYN", True, True,
                              False, False, False, True)
        record.validate()
        for flag in ("historical_truth_allowed", "alice_lived_memory", "autobiographical_recall_allowed"):
            with self.assertRaises(IdentityError):
                replace(record, **{flag: True}).validate()

    def test_context_only_graph_edge_cannot_enter_identity(self):
        graph = replace(self.frame.graph, roles=(("context_only", *self.frame.graph.roles[0][1:]),
                                                 self.frame.graph.roles[1]))
        with self.assertRaisesRegex(IdentityError, "context-only support"):
            self.model(replace(self.frame, graph=graph))
        graph = replace(self.frame.graph, receivers=self.frame.graph.receivers.clone())
        graph.receivers[0, 0] = 0
        with self.assertRaisesRegex(IdentityError, "without rewriting sources"):
            self.model(replace(self.frame, graph=graph))

    def test_missing_identity_exposes_insufficiency_not_confidence(self):
        records = tuple(tuple(replace(record, identity_core_allowed=False) for record in row)
                        for row in self.frame.source_records)
        roles = tuple(tuple("excluded_context" if role in {"identity_support", "inference_support"} else role
                            for role in row) for row in self.frame.graph.roles)
        packet = self.model(replace(self.frame, source_records=records, graph=replace(self.frame.graph, roles=roles)))
        self.assertFalse(packet.identity_grounding_available.any())
        self.assertTrue((packet.evidence_sufficiency == 0).all())
        self.assertTrue((packet.uncertainty == 1).all())
        self.assertTrue((packet.voice_control_confidence == 0).all())
        self.assertTrue((packet.representation.views["identity"] == 0).all())

    def test_no_available_candidate_has_zero_mass_and_no_fabricated_decision(self):
        frame = replace(self.frame, candidate_context_mask=torch.zeros_like(self.frame.candidate_context_mask))
        packet = self.model(frame)
        self.assertFalse(packet.candidate_mask.any())
        self.assertTrue((packet.preferences == 0).all())
        self.assertTrue((packet.latent_candidates == 0).all())
        self.assertTrue((packet.uncertainty == 1).all())
        self.assertTrue(torch.isfinite(packet.preferences).all())

    def test_singleton_preference_is_not_evidence_certainty(self):
        packet = self.model(_frame(self.config, candidates=1))
        self.assertTrue((packet.preferences == 1).all())
        self.assertTrue(((packet.uncertainty > 0) & (packet.uncertainty < 1)).all())
        self.assertTrue(((packet.evidence_sufficiency > 0) & (packet.evidence_sufficiency < 1)).all())

    def test_dynamic_candidates_labels_and_token_lengths_are_not_fixed_classes(self):
        frame = _frame(self.config, batch=1, candidates=11)
        labels, specs = dict(frame.labels), dict(frame.label_specs)
        labels["values"] = _bank(1, 7, self.config, "expanded-value", tokens=9)
        specs["values"] = (tuple(LabelSpec(f"expanded-value-{i}", f"Value {i}", f"Runtime semantic value {i}")
                                   for i in range(7)),)
        packet = self.model(replace(frame, labels=labels, label_specs=specs))
        self.assertEqual(packet.preferences.shape, (1, 11))
        self.assertEqual(packet.heads["values"].scores.shape, (1, 11, 7))
        self.assertEqual(packet.value_tradeoff_margins.shape, (1, 11, 7, 7))

    def test_dynamic_source_concept_relation_graph_and_directed_lineage(self):
        sources = _bank(2, 7, self.config, "source")
        concepts = _bank(2, 6, self.config, "concept")
        relations = _bank(2, 3, self.config, "relation")
        records = tuple(row + tuple(SourceRecord(f"source-{i}", f"family-{i}", "E0", "E0",
                                                 True, False, True, False, False, False)
                                     for i in (5, 6)) for row in self.frame.source_records)
        concept_records = tuple(row + (ConceptRecord("concept-5", "identity", True),)
                                for row in self.frame.concept_records)
        graph = self.frame.graph
        graph = SupportGraph(torch.cat((graph.senders, torch.full((2, 1), 7)), 1),
                             torch.cat((graph.receivers + 2, torch.full((2, 1), 12)), 1),
                             torch.cat((graph.relation_indices, torch.full((2, 1), 2)), 1),
                             torch.ones(2, 7, dtype=torch.bool),
                             tuple(row + ("identity_support",) for row in graph.roles))
        packet = self.model(replace(self.frame, sources=sources, concepts=concepts, relations=relations,
                                    source_records=records, concept_records=concept_records, graph=graph))
        self.assertEqual(packet.evidence_pointers.shape, (2, 3, 7))
        self.assertEqual(packet.concept_pointers.shape, (2, 3, 6))
        self.assertTrue(packet.representation.concept_mask[:, 5].all())
        self.assertTrue(packet.representation.concept_source_mask[:, 5, 0].all())
        self.assertFalse(packet.representation.concept_source_mask[:, 5, 2:5].any())

    def test_pointer_ids_do_not_define_learned_semantics(self):
        original = self.model(self.frame)
        sources = replace(self.frame.sources, entry_ids=tuple(tuple(f"new-source-{i}" for i in range(5)) for _ in range(2)))
        records = tuple(tuple(replace(record, record_id=f"new-source-{i}", source_family_id=f"new-family-{i}")
                              for i, record in enumerate(row)) for row in self.frame.source_records)
        candidates = replace(self.frame.candidates, entry_ids=(("random-c", "random-a", "random-b"),) * 2)
        changed = self.model(replace(self.frame, sources=sources, source_records=records, candidates=candidates))
        torch.testing.assert_close(original.preferences, changed.preferences)
        torch.testing.assert_close(original.heads["values"].scores, changed.heads["values"].scores)
        self.assertEqual(changed.candidate_ids, candidates.entry_ids)

    def test_padding_tokens_cannot_change_source_identity(self):
        token_mask = self.frame.sources.token_mask.clone()
        token_mask[:, :, -1] = False
        source = replace(self.frame.sources, token_mask=token_mask)
        frame = replace(self.frame, sources=source)
        original = self.model(frame)
        changed_states = source.states.clone()
        changed_states[:, :, :, -1] += 1000
        changed = self.model(replace(frame, sources=replace(source, states=changed_states)))
        torch.testing.assert_close(original.preferences, changed.preferences)
        torch.testing.assert_close(original.representation.views["identity"], changed.representation.views["identity"])

    def test_strict_flags_geometry_and_complete_heads_refuse_unsupported_input(self):
        records = [list(row) for row in self.frame.source_records]
        records[0][0] = replace(records[0][0], identity_core_allowed="false")
        with self.assertRaisesRegex(IdentityError, "actual boolean"):
            self.model(replace(self.frame, source_records=tuple(tuple(row) for row in records)))
        with self.assertRaisesRegex(IdentityError, "actual provider geometry"):
            self.model(replace(self.frame, sources=replace(self.frame.sources, states=self.frame.sources.states[:, :, :1])))
        labels = dict(self.frame.labels)
        labels.pop("voice")
        with self.assertRaisesRegex(IdentityError, "multi-head label families"):
            self.model(replace(self.frame, labels=labels))
        voice = [list(row) for row in self.frame.label_specs["voice"]]
        voice[0][0] = replace(voice[0][0], unit="categorical")
        specs = dict(self.frame.label_specs)
        specs["voice"] = tuple(tuple(row) for row in voice)
        with self.assertRaisesRegex(IdentityError, "portable units"):
            self.model(replace(self.frame, label_specs=specs))

    def test_n3_gradients_are_calibration_only_and_never_rewrite_sources(self):
        batch = CalibrationBatch(self.frame, {name: torch.full((2, 3), .5) for name in
                                  ("uncertainty", "failure_tail_risk", "owner_fidelity_uncertainty")},
                                 torch.ones(2, 3, dtype=torch.bool), "public_mechanical_fixture",
                                 "public-random-tensor-family", "a" * 64)
        with self.assertRaisesRegex(IdentityError, "supplied calibration data"):
            self.model.set_phase("calibration")
        with self.assertRaisesRegex(IdentityError, "explicit admitted calibration phase"):
            self.model.calibration_loss(batch)
        self.model.set_phase("calibration", calibration_batch=batch)
        self.model.calibration_loss(batch).backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in self.model.n3.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for module in (self.model.n1, self.model.n2)
                            for p in module.parameters()))
        self.assertTrue(all(record.historical_truth_allowed == (col == 0)
                            for row in self.frame.source_records for col, record in enumerate(row)))
        self.model.set_phase("n2")
        self.assertTrue(all(not p.requires_grad for p in self.model.n3.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.model.n1.parameters()))
        self.model.set_phase("inference")
        with self.assertRaisesRegex(IdentityError, "explicit learned or calibration phase"):
            self.model.train()

    def test_sparse_n3_targets_preserve_missing_review_without_placeholders(self):
        availability = torch.zeros(2, 3, dtype=torch.bool)
        availability[:, 0] = True
        targets = torch.full((2, 3), float("nan"))
        targets[:, 0] = .5
        batch = CalibrationBatch(self.frame, {"uncertainty": targets}, availability,
                                 "public_mechanical_fixture", "sparse-public-family", "b" * 64,
                                 target_masks={"uncertainty": availability.clone()})
        self.model.set_phase("calibration", calibration_batch=batch)
        loss = self.model.calibration_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.n3.parameters()))
        with self.assertRaisesRegex(IdentityError, "calibration frame"):
            self.model(replace(self.frame))
        wrong_mask = availability.clone()
        wrong_mask[0, 0] = False
        with self.assertRaisesRegex(IdentityError, "availability union"):
            replace(batch, target_masks={"uncertainty": wrong_mask}).validate(self.config)
        with self.assertRaisesRegex(IdentityError, "separate calibration data"):
            replace(batch, split="TRAIN").validate(self.config)

    def test_nonfinite_learned_packet_is_refused(self):
        with torch.no_grad():
            self.model.n2.scalar[-1].weight[0, 0] = float("nan")
        with self.assertRaisesRegex(IdentityError, "nonfinite learned outputs"):
            self.model(self.frame)

    def test_save_reload_binds_core_and_separate_calibration_without_approval(self):
        with torch.no_grad():
            self.model.n3.log_temperatures["temperature_preference"].add_(.2)
        original = self.model(self.frame)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "snapshot"
            manifest = save_untrained_snapshot(self.model, path)
            self.assertEqual(manifest["training_state"], "UNTRAINED")
            self.assertEqual(manifest["qualification"], "UNQUALIFIED")
            self.assertIsNone(manifest["training_evidence"])
            self.assertFalse(manifest["source_acceptance_authority"])
            self.assertTrue(manifest["calibration_separate_from_core"])
            core = torch.load(path / "core.pt", weights_only=True)
            self.assertFalse(any(name.startswith("n3.") for name in core))
            reloaded = load_untrained_snapshot(path)
            changed = reloaded(self.frame)
            torch.testing.assert_close(original.preferences, changed.preferences)
            torch.testing.assert_close(original.uncertainty, changed.uncertainty)
            self.assertTrue(all(not p.requires_grad for p in reloaded.parameters()))
            with self.assertRaisesRegex(IdentityError, "new absolute directory"):
                save_untrained_snapshot(self.model, path)
            payload = (path / "core.pt").read_bytes()
            (path / "core.pt").write_bytes(payload + b"tampered")
            with self.assertRaisesRegex(IdentityError, "learned-state hash"):
                load_untrained_snapshot(path)


if __name__ == "__main__":
    unittest.main()
