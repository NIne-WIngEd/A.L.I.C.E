"""Random public mechanics only; fixtures are not reviewed Alice identity gold.

These tests exercise real CPU Torch losses/AdamW/continuation. Fixture custody
artifacts explicitly identify their synthetic limitation and grant no private
gradient or behavior acceptance. They never open the real source packages.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from src.alice_personality.identity import (HEAD_FAMILIES, ConceptRecord, FrozenTokenBank,
    IdentityError, IdentityFrame, IdentityModel, IdentityModelConfig, LabelSpec, SourceRecord, SupportGraph)
from src.alice_personality.identity.contracts import SOURCE_KINDS
from src.alice_personality.teaching import (ArtifactPin, ContrastConstraint, IdentityObjectives, IdentityTrainer,
    ReviewedTarget, TARGET_SCHEMA, TeachingAdmission, TeachingBatch, TeachingError, TrainingRecipe,
    current_code_hashes, model_fingerprint)
from src.alice_personality.teaching.contracts import (AUTH_SCHEMA, FAMILY_SCHEMA, REVIEW_SCHEMA,
    canonical, fingerprint)
from src.alice_personality.teaching.trainer import _production


def _bank(count, config, prefix):
    states = torch.randn(1, count, config.provider_state_count, 3, config.provider_width)
    mask = torch.ones(1, count, 3, dtype=torch.bool)
    return FrozenTokenBank(states, mask, mask.any(-1), (tuple(f"{prefix}-{i}" for i in range(count)),))


def _frame(config, family="train"):
    query, sources, concepts = _bank(1, config, "query"), _bank(4, config, "source"), _bank(4, config, "concept")
    candidates, relations = _bank(3, config, "candidate"), _bank(1, config, "relation")
    records = ((SourceRecord("source-0", family, "E0", "E0", True, True, True, False, False, False),
                SourceRecord("source-1", family, "EINF", "E-INF", True, True, False, False, False, False),
                SourceRecord("source-2", family, "HOST", "HOST", False, True, False, False, False, False),
                SourceRecord("source-3", family, "AEXP", "A-EXP", False, True, False, True, True, False)),)
    views = ("identity", "host", "self", "context")
    concept_records = (tuple(ConceptRecord(f"concept-{i}", view, residual=i == 0) for i, view in enumerate(views)),)
    labels = {name: _bank(3, config, name) for name in HEAD_FAMILIES}
    specs = {name: (tuple(LabelSpec(f"{name}-{i}", f"fictional {name} label {i}", "random public mechanical label",
                                   "words_per_sentence" if name == "voice" else "categorical",
                                   3.0 if name == "voice" else None, 23.0 if name == "voice" else None)
                          for i in range(3)),) for name in HEAD_FAMILIES}
    graph = SupportGraph(torch.tensor([[0, 1, 2, 3, 1]]), torch.tensor([[4, 4, 5, 6, 7]]),
        torch.zeros(1, 5, dtype=torch.int64), torch.ones(1, 5, dtype=torch.bool),
        (("identity_support", "inference_support", "context_only", "context_only", "context_only"),))
    mask = torch.ones(1, 3, dtype=torch.bool)
    return IdentityFrame(query, sources, concepts, candidates, relations, labels, records, concept_records,
                         specs, graph, mask.clone(), mask.clone())


def _target(values, mask=None, uncertainty=0.0):
    values = values.detach().clone().float() if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float32)
    mask = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.clone()
    uncertainty = torch.full_like(values, uncertainty) if not isinstance(uncertainty, torch.Tensor) else uncertainty.clone()
    return ReviewedTarget(values, mask, uncertainty)


def _batch(model, frame, phase="n2", split="TRAIN", name="case", *, full=False):
    if phase == "n1":
        packet = model(frame)  # Geometry/masks only; no generated behavioral teaching facts.
        mask = packet.representation.concept_source_mask
        distribution = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        distribution[:, 0, :2] = torch.tensor([0.75, 0.25])
        provenance = torch.zeros(1, 4, len(SOURCE_KINDS))
        for i, record in enumerate(frame.source_records[0]):
            provenance[0, i, SOURCE_KINDS.index(record.source_kind)] = 1
        targets = {"concept_source_alignment": _target(distribution, mask), "provenance": _target(provenance)}
    elif phase == "calibration":
        # Sparse independent owner target, with unavailable facts carried as NaN.
        targets = {"owner_fidelity_uncertainty": _target([[0.3, float("nan"), 0.8]],
                                                       torch.tensor([[True, False, True]]))}
    else:
        targets = {"preferences": _target([[0.4, 0.4, 0.2]]), "co_valid_probabilities": _target([[1.0, 1.0, 0.0]])}
        if full:
            targets.update({name: _target([[0.2, 0.4, 0.6]]) for name in
                ("evidence_sufficiency", "uncertainty", "contraindications", "failure_tail_risk")})
            for name in HEAD_FAMILIES:
                if name != "voice":
                    values = [[[0.2, 0.3, 0.5]] * 3] if name == "stance" else [[[1.0, 1.0, 0.0]] * 3]
                    targets["head:" + name] = _target(values)
            targets["voice_control_values"] = _target([[[8.0, 13.0, 18.0]] * 3])
            targets["voice_control_confidence"] = _target([[[0.8, 0.7, 0.6]] * 3])
            targets["evidence_pointers"] = _target([[[0.4, 0.3, 0.2, 0.1]] * 3])
            historical = torch.tensor([[[True, False, False, False]] * 3])
            experience = torch.tensor([[[False, False, False, True]] * 3])
            targets["historical_evidence_pointers"] = _target([[[1.0, 0, 0, 0]] * 3], historical)
            targets["alice_experience_pointers"] = _target([[[0, 0, 0, 1.0]] * 3], experience)
            targets["concept_pointers"] = _target([[[0.25, 0.25, 0.25, 0.25]] * 3])
            pair_mask = ~torch.eye(3, dtype=torch.bool)[None, None].expand(1, 3, -1, -1)
            targets["value_tradeoff_probabilities"] = _target(torch.tensor([[[[0.5, 0.7, 0.8],
                [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]]] * 3]), pair_mask)
    families = (tuple(sorted({item.source_family_id for item in frame.source_records[0]})),)
    return TeachingBatch(frame, targets, (name,), families, phase, split, "public_mechanical_fixture")


def _admission(directory, model, recipe, batches, *, auth_changes=None, reviewed_changes=None, family_changes=None):
    directory = Path(directory)
    artifacts = {}
    def write(role, data):
        path = directory / (role + ".json")
        payload = data if isinstance(data, bytes) else canonical(data) + b"\n"
        path.write_bytes(payload)
        artifacts[role] = ArtifactPin(str(path), sha256(payload).hexdigest())
    write("source_package", b"public mechanical source bytes; no identity payload")
    write("compiled_receipt", {"schema": "public-mechanical-source-receipt-v1",
                               "training_authority_granted": False, "acceptance_authority": False})
    write("prepared_receipt", {"schema": "public-mechanical-feature-receipt-v1", "qualification": "UNQUALIFIED",
                               "model_geometry": {"hidden_size": model.config.provider_width,
                                                  "hidden_state_count": model.config.provider_state_count}})
    review = {"schema": REVIEW_SCHEMA, "target_schema": TARGET_SCHEMA, "reviewed": True,
              "source_class": "public_mechanical_fixture", "reviewed_batches": {}, "batch_order": {}}
    splits = {}
    for batch in batches:
        digest = batch.fingerprint()
        review["reviewed_batches"][digest] = {"split": batch.split, "case_ids": list(batch.case_ids),
                                               "source_family_ids": [list(row) for row in batch.source_family_ids]}
        review["batch_order"].setdefault(batch.split, []).append(digest)
        for row in batch.source_family_ids:
            for family in row:
                splits[family] = batch.split
    review.update(reviewed_changes or {})
    write("reviewed_targets", review)
    splits.update(family_changes or {})
    write("source_families", {"schema": FAMILY_SCHEMA, "family_splits": splits})
    config_hash, initial = sha256(canonical(asdict(model.config))).hexdigest(), model_fingerprint(model)
    code = current_code_hashes()
    auth = {"schema": AUTH_SCHEMA, "authorization_id": "public-implementation-test-only", "phase": recipe.phase,
            "source_class": "public_mechanical_fixture", "private_gradient_authorized": False,
            "mechanical_steps_authorized": True, "artifact_file_sha256": {key: pin.sha256 for key, pin in artifacts.items()},
            "model_config_sha256": config_hash, "model_initial_state_sha256": initial,
            "recipe_sha256": recipe.fingerprint(), "code_sha256": code}
    auth.update(auth_changes or {})
    write("authorization", auth)
    return TeachingAdmission(artifacts, config_hash, initial, recipe.fingerprint(), code)


def _production_fixture(frame):
    """Only a public metadata validation fixture, never real custody evidence."""
    code = {name: current_code_hashes()["identity/" + name] for name in ("codec.py", "feature_producer.py")}
    common = {"source_class": "prepared_frozen_gemma_features_unqualified", "prepared_receipt_file_sha256": "a" * 64,
              "implementation_sha256": code, "qualification": "UNQUALIFIED", "source_acceptance_authority": False,
              "persistent_feature_values_retained": False}
    binding = {**common, "schema": "alice-personality-produced-frame-binding-v1", "frame_sha256": fingerprint(frame),
               "document_sha256": "d" * 64, "source_geometry": [frame.query.states.shape[2], frame.query.states.shape[-1]],
               "session_source_recheck_pending": True, "chat_template": False, "truncation": False, "generation": False}
    binding["binding_sha256"] = sha256(canonical(binding)).hexdigest()
    session = {**common, "schema": "alice-personality-feature-session-v1", "prepared_receipt_sha256": "b" * 64,
               "status": "CLOSED_SOURCE_VERIFIED", "session_source_recheck_passed": True, "mechanical_fixture_only": False,
               "frame_binding_sha256": [binding["binding_sha256"]]}
    session["receipt_sha256"] = sha256(canonical(session)).hexdigest()
    kwargs = {"prepared_file_sha256": "a" * 64, "prepared_receipt_sha256": "b" * 64, "implementation": code}
    return binding, session, kwargs


class IdentityTeachingMechanicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(8411)
        self.config = IdentityModelConfig(4, 3, learned_width=8, attention_heads=2, graph_layers=1, readout_layers=1)
        self.frame = _frame(self.config)
        self.model = IdentityModel(self.config)
        self.initial = copy.deepcopy(self.model.state_dict())
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.objectives = IdentityObjectives(self.config.learned_width)

    def tearDown(self):
        self.temporary.cleanup()

    def trainer(self, phase="n2", *, full=False, batches=None, adapt_n1=False, **admission_changes):
        recipe = TrainingRecipe(phase, learning_rate=0.01, max_grad_norm=0.5, seed=777, adapt_n1=adapt_n1)
        split = "CALIBRATION_TRAIN" if phase == "calibration" else "TRAIN"
        batch = _batch(self.model, self.frame, phase, split, full=full)
        batches = batches or [batch]
        admission = _admission(self.path, self.model, recipe, batches, **admission_changes)
        return IdentityTrainer(self.model, recipe, admission, first_batch=batches[0]), batches[0]

    def test_n2_full_packet_has_real_losses_firstparty_updates_and_frozen_n1_n3(self):
        trainer, batch = self.trainer(full=True)
        before = copy.deepcopy(self.model.state_dict())
        for bank in (batch.frame.query, batch.frame.sources, batch.frame.candidates, *batch.frame.labels.values()):
            bank.states.requires_grad_(True)
        event = trainer.step(batch)
        self.assertEqual(set(event["objectives"]), set(batch.targets))
        self.assertTrue(all(np.isfinite(value) for value in event["objectives"].values()))
        after = self.model.state_dict()
        self.assertTrue(any(not torch.equal(before[key], after[key]) for key in before if key.startswith("n2.")))
        self.assertTrue(all(torch.equal(before[key], after[key]) for key in before if key.startswith(("n1.", "n3."))))
        self.assertTrue(all(p.grad is None for p in self.model.n1.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.n3.parameters()))
        self.assertIsNone(batch.frame.sources.states.grad)
        self.assertEqual(event["qualification"], "UNQUALIFIED")
        self.assertIsNone(event["behavior_qualification"])
        self.assertFalse(event["acceptance_authority"])
        self.assertFalse(event["private_gradient_authorized"])

    def test_n1_alignment_and_provenance_update_only_n1_and_teaching_auxiliary(self):
        trainer, batch = self.trainer("n1")
        before, auxiliary = copy.deepcopy(self.model.state_dict()), copy.deepcopy(trainer.objectives.state_dict())
        event = trainer.step(batch)
        self.assertEqual(set(event["objectives"]), {"concept_source_alignment", "provenance"})
        self.assertTrue(any(not torch.equal(before[k], self.model.state_dict()[k]) for k in before if k.startswith("n1.")))
        self.assertTrue(all(torch.equal(before[k], self.model.state_dict()[k]) for k in before if k.startswith(("n2.", "n3."))))
        self.assertTrue(any(not torch.equal(auxiliary[k], trainer.objectives.state_dict()[k]) for k in auxiliary))

    def test_n3_sparse_owner_targets_update_only_separate_calibration(self):
        trainer, batch = self.trainer("calibration")
        before = copy.deepcopy(self.model.state_dict())
        event = trainer.step(batch)
        self.assertEqual(set(event["objectives"]), {"owner_fidelity_uncertainty"})
        self.assertEqual(event["reviewed_target_coverage"]["owner_fidelity_uncertainty"], 2)
        self.assertTrue(any(not torch.equal(before[k], self.model.state_dict()[k]) for k in before if k.startswith("n3.")))
        self.assertTrue(all(torch.equal(before[k], self.model.state_dict()[k]) for k in before if k.startswith(("n1.", "n2."))))

    def test_n2_explicit_adaptation_permits_n1_continuation(self):
        trainer, batch = self.trainer(adapt_n1=True)
        before = copy.deepcopy(self.model.n1.state_dict())
        trainer.step(batch)
        self.assertTrue(any(not torch.equal(before[k], self.model.n1.state_dict()[k]) for k in before))
        self.assertTrue(all(p.grad is None for p in self.model.n3.parameters()))

    def test_manual_optimizer_source_parameter_injection_is_rejected(self):
        trainer, batch = self.trainer()
        trainer.optimizer.add_param_group({"params": [torch.nn.Parameter(torch.randn(4))]})
        with self.assertRaisesRegex(TeachingError, "firstparty stage"):
            trainer.step(batch)
        self.assertEqual(trainer.steps, 0)

    def test_optimizer_recipe_tamper_is_rejected(self):
        trainer, batch = self.trainer()
        trainer.optimizer.param_groups[0]["lr"] *= 10
        with self.assertRaisesRegex(TeachingError, "hyperparameters"):
            trainer.step(batch)

    def test_nonfinite_gradient_prevents_update_and_poisoned_state_cannot_continue(self):
        trainer, batch = self.trainer()
        handles = [parameter.register_hook(lambda grad: grad * float("inf")) for parameter in self.model.n2.parameters()]
        with self.assertRaisesRegex(TeachingError, "nonfinite"):
            trainer.step(batch)
        for handle in handles:
            handle.remove()
        self.assertTrue(trainer.failed)
        self.assertEqual(trainer.steps, 0)
        with self.assertRaisesRegex(TeachingError, "failed optimization"):
            trainer.step(batch)
        with self.assertRaises(TeachingError):
            trainer.save_checkpoint(self.path / "failed")

    def test_masked_missing_target_cannot_become_a_negative_or_gradient(self):
        batch = _batch(self.model, self.frame)
        packet = self.model(self.frame)
        scores = torch.tensor([[0.5, -0.2, 8.0]], requires_grad=True)
        packet = replace(packet, preference_logits=scores)
        available = torch.tensor([[True, True, False]])
        target = _target([[0.5, 0.5, float("nan")]], available)
        loss, _, _ = self.objectives(packet, replace(batch, targets={"preferences": target}))
        loss.backward()
        self.assertEqual(float(scores.grad[0, 2]), 0)
        changed = replace(packet, preference_logits=scores.detach() + torch.tensor([[0.0, 0.0, 1000.0]]))
        second, _, _ = self.objectives(changed, replace(batch, targets={"preferences": target}))
        torch.testing.assert_close(loss.detach(), second)

    def test_multilabel_co_validity_does_not_force_one_winner(self):
        batch = _batch(self.model, self.frame)
        values = torch.tensor([[0.4, 0.6, 0.5]], requires_grad=True)
        packet = replace(self.model(self.frame), co_valid_probabilities=values)
        loss, _, _ = self.objectives(packet, replace(batch, targets={"co_valid_probabilities": _target([[1.0, 1.0, 0]])}))
        loss.backward()
        self.assertLess(float(values.grad[0, 0]), 0)
        self.assertLess(float(values.grad[0, 1]), 0)
        self.assertGreater(float(values.grad[0, 2]), 0)

    def test_uncertainty_one_removes_only_its_reviewed_loss(self):
        batch = _batch(self.model, self.frame)
        values = torch.tensor([[0.4, 0.6, 0.5]], requires_grad=True)
        packet = replace(self.model(self.frame), co_valid_probabilities=values)
        target = _target([[1.0, 1.0, 0]], uncertainty=torch.tensor([[0.0, 1.0, 0.0]]))
        loss, _, _ = self.objectives(packet, replace(batch, targets={"co_valid_probabilities": target}))
        loss.backward()
        self.assertEqual(float(values.grad[0, 1]), 0)
        self.assertNotEqual(float(values.grad[0, 0]), 0)

    def test_candidate_permutation_with_explicit_targets_preserves_objective(self):
        batch = _batch(self.model, self.frame)
        packet = self.model(self.frame)
        first, _, _ = self.objectives(packet, batch)
        order = [2, 0, 1]
        changed_packet = replace(packet, preference_logits=packet.preference_logits[:, order],
                                 co_valid_probabilities=packet.co_valid_probabilities[:, order],
                                 candidate_mask=packet.candidate_mask[:, order])
        changed_targets = {key: ReviewedTarget(value.values[:, order], value.available[:, order], value.uncertainty[:, order])
                           for key, value in batch.targets.items()}
        second, _, _ = self.objectives(changed_packet, replace(batch, targets=changed_targets))
        torch.testing.assert_close(first, second)

    def test_voice_targets_use_declared_portable_range_and_refuse_out_of_range(self):
        batch = _batch(self.model, self.frame, full=True)
        packet = self.model(self.frame)
        target = batch.targets["voice_control_values"]
        total, _, _ = self.objectives(packet, replace(batch, targets={"voice_control_values": target}))
        self.assertTrue(torch.isfinite(total))
        with self.assertRaisesRegex(TeachingError, "portable-unit ranges"):
            self.objectives(packet, replace(batch, targets={"voice_control_values": _target(torch.full((1, 3, 3), 30.0))}))

    def test_identity_alignment_cannot_use_host_sources(self):
        batch = _batch(self.model, self.frame, "n1")
        target = batch.targets["concept_source_alignment"]
        available = target.available.clone()
        available[:, 0, 2] = True
        with self.assertRaisesRegex(TeachingError, "unauthorized output"):
            self.objectives(self.model(self.frame), replace(batch, targets={"concept_source_alignment": replace(target, available=available)}))

    def test_historical_and_alice_experience_pointers_keep_distinct_authority(self):
        batch = _batch(self.model, self.frame)
        target = _target([[[0, 0, 1.0, 0]] * 3], torch.tensor([[[False, False, True, False]] * 3]))
        for name in ("historical_evidence_pointers", "alice_experience_pointers"):
            with self.subTest(name=name), self.assertRaisesRegex(TeachingError, "unauthorized output"):
                self.objectives(self.model(self.frame), replace(batch, targets={name: target}))

    def test_contradictory_reviewed_provenance_is_rejected(self):
        batch = _batch(self.model, self.frame, "n1")
        values = batch.targets["provenance"].values.clone()
        values[:, 0] = 0
        values[:, 0, SOURCE_KINDS.index("HOST")] = 1
        with self.assertRaisesRegex(TeachingError, "contradicts"):
            self.objectives(self.model(self.frame), replace(batch, targets={"provenance": _target(values)}))

    def test_missing_reviewed_facts_or_malformed_distribution_are_not_filled(self):
        batch = _batch(self.model, self.frame)
        with self.assertRaisesRegex(TeachingError, "source rows"):
            replace(batch, targets={}).validate(self.config)
        with self.assertRaisesRegex(TeachingError, "sum to one"):
            self.objectives(self.model(self.frame), replace(batch, targets={"preferences": _target([[1.0, 1.0, 0]])}))
        with self.assertRaisesRegex(TeachingError, "not implemented"):
            self.objectives(self.model(self.frame), replace(batch, targets={"unordered_alternative_is_negative": _target([[1., 1., 1.]])}))

    def test_strict_availability_and_available_nonfinite_are_refused(self):
        batch = _batch(self.model, self.frame)
        for target in (_target([[0.2, float("nan"), 0.8]]),
                       replace(_target([[0.2, 0.3, 0.5]]), available=torch.ones(1, 3))):
            with self.subTest(target=type(target.available)), self.assertRaises(TeachingError):
                self.objectives(self.model(self.frame), replace(batch, targets={"preferences": target}))

    def test_explicit_contrast_relevance_drives_only_declared_metric(self):
        batch = _batch(self.model, self.frame)
        left = replace(self.model(self.frame), co_valid_probabilities=torch.tensor([[0.2, 0.3, 0.4]], requires_grad=True))
        right = replace(self.model(self.frame), co_valid_probabilities=torch.tensor([[0.3, 0.4, 0.9]]))
        constraint = ContrastConstraint(self.frame, "co_valid_probabilities", "relevant", torch.tensor([[True, True, False]]),
                                        torch.tensor([True]), torch.tensor([0.5]), torch.tensor([0.0]))
        loss = self.objectives.contrast_loss(left, right, constraint, batch)
        torch.testing.assert_close(loss, torch.tensor(0.4))
        self.assertEqual(float(self.objectives.contrast_loss(left, right, replace(constraint, relation="irrelevant"), batch).detach()), 0)
        with self.assertRaisesRegex(TeachingError, "explicit relevance"):
            self.objectives.contrast_loss(left, right, replace(constraint, relation="unordered"), batch)

    def test_owner_fidelity_gold_is_separate_from_n1_n2_teaching(self):
        batch = _batch(self.model, self.frame)
        with self.assertRaisesRegex(TeachingError, "not implemented"):
            self.objectives(self.model(self.frame), replace(batch, targets={"owner_fidelity_uncertainty": _target([[0.1, 0.2, 0.3]])}))

    def test_absent_identity_grounding_cannot_be_taught_as_certainty(self):
        records = tuple(tuple(replace(record, identity_core_allowed=False) for record in row) for row in self.frame.source_records)
        roles = tuple(tuple("excluded_context" if role in {"identity_support", "inference_support"} else role
                            for role in row) for row in self.frame.graph.roles)
        frame = replace(self.frame, source_records=records, graph=replace(self.frame.graph, roles=roles))
        batch, packet = _batch(self.model, frame), self.model(frame)
        self.assertFalse(packet.identity_grounding_available.any())
        for name, invalid in (("uncertainty", 0.0), ("evidence_sufficiency", 1.0), ("failure_tail_risk", 0.0)):
            with self.subTest(name=name), self.assertRaisesRegex(TeachingError, "absent identity grounding"):
                self.objectives(packet, replace(batch, targets={name: _target([[invalid] * 3])}))

    def test_reviewed_value_tradeoffs_ignore_unknown_pairs_and_reject_contradiction(self):
        batch = _batch(self.model, self.frame, full=True)
        target = batch.targets["value_tradeoff_probabilities"]
        packet = self.model(self.frame)
        margins = packet.value_tradeoff_margins.detach().clone().requires_grad_(True)
        packet = replace(packet, value_tradeoff_margins=margins)
        mask = torch.zeros_like(target.available)
        mask[..., 0, 1] = True
        values = torch.full_like(target.values, float("nan"))
        values[..., 0, 1] = 0.8
        partial = replace(target, values=values, available=mask)
        loss, _, _ = self.objectives(packet, replace(batch, targets={"value_tradeoff_probabilities": partial}))
        loss.backward()
        self.assertTrue((margins.grad[~mask] == 0).all())
        contradictory = target.values.clone()
        contradictory[..., 1, 0] = 0.7
        with self.assertRaisesRegex(TeachingError, "complementary"):
            self.objectives(packet, replace(batch, targets={"value_tradeoff_probabilities": replace(target, values=contradictory)}))
        with self.assertRaisesRegex(TeachingError, "unauthorized output"):
            self.objectives(packet, replace(batch, targets={"value_tradeoff_probabilities": replace(target, available=torch.ones_like(mask))}))

    def test_current_authorization_does_not_come_from_pending_compiler_flags(self):
        trainer, batch = self.trainer()
        self.assertFalse(json.loads(Path(trainer.admission.artifacts["compiled_receipt"].path).read_text())["training_authority_granted"])
        self.assertEqual(trainer.step(batch)["step"], 1)

    def test_fake_truthy_authorization_or_private_fixture_grant_is_refused(self):
        for changes in ({"mechanical_steps_authorized": "true"}, {"private_gradient_authorized": True},
                        {"phase": "n1"}, {"artifact_file_sha256": {}}):
            with self.subTest(changes=changes), self.assertRaises(TeachingError):
                self.trainer(auth_changes=changes)

    def test_compiler_rows_without_review_receipt_are_refused(self):
        with self.assertRaisesRegex(TeachingError, "not a reviewed"):
            self.trainer(reviewed_changes={"schema": "alice-eipm-governed-source-compilation-v1"})

    def test_external_source_code_model_recipe_and_feature_geometry_pins_are_enforced(self):
        trainer, batch = self.trainer()
        admission = trainer.admission
        changed = [replace(admission, code_sha256={}), replace(admission, model_config_sha256="0" * 64),
                   replace(admission, recipe_sha256="0" * 64), replace(admission, model_initial_state_sha256="0" * 64)]
        for value in changed:
            with self.subTest(value=value.model_initial_state_sha256), self.assertRaises(TeachingError):
                value.verify(self.model, trainer.recipe, initial=True)
        source = Path(admission.artifacts["source_package"].path)
        source.write_bytes(b"changed public mechanical source")
        with self.assertRaisesRegex(TeachingError, "external SHA pin"):
            trainer.step(batch)
        self.assertEqual(trainer.steps, 0)

    def test_target_or_frozen_feature_tamper_after_review_is_refused(self):
        trainer, batch = self.trainer()
        changed = replace(batch, targets={"preferences": _target([[0.3, 0.3, 0.4]])})
        with self.assertRaisesRegex(TeachingError, "externally reviewed"):
            trainer.step(changed)
        batch.frame.sources.states[:, 0, 0, 0, 0] += 1
        with self.assertRaisesRegex(TeachingError, "externally reviewed"):
            trainer.step(batch)

    def test_shared_families_and_final_split_cannot_cross_holdouts(self):
        with self.assertRaisesRegex(TeachingError, "held-out"):
            self.trainer(family_changes={"train": "DEV"})
        batch = _batch(self.model, self.frame)
        with self.assertRaisesRegex(TeachingError, "FINAL"):
            replace(batch, split="FINAL").validate(self.config)

    def test_dev_evidence_is_separate_and_cannot_optimize(self):
        train = _batch(self.model, self.frame)
        dev = _batch(self.model, _frame(self.config, "dev"), split="DEV", name="dev-case")
        trainer, _ = self.trainer(batches=[train, dev])
        before = copy.deepcopy(self.model.state_dict())
        event = trainer.evaluate(dev)
        self.assertEqual(event["split"], "DEV")
        self.assertFalse(event["acceptance_authority"])
        self.assertTrue(all(torch.equal(before[k], self.model.state_dict()[k]) for k in before))
        self.assertEqual(trainer.steps, 0)
        with self.assertRaisesRegex(TeachingError, "replay order"):
            trainer.step(dev)

    def test_explicit_replay_order_is_required(self):
        a = _batch(self.model, self.frame, name="first")
        b = replace(a, case_ids=("second",))
        trainer, _ = self.trainer(batches=[a, b])
        with self.assertRaisesRegex(TeachingError, "replay order"):
            trainer.step(b)
        trainer.step(a)
        trainer.step(b)
        self.assertEqual((trainer.steps, trainer.epoch, trainer.batch_index), (2, 1, 0))

    def test_resumable_optimizer_rng_and_cursor_reproduce_exact_next_update(self):
        a = _batch(self.model, self.frame, name="first")
        b = replace(a, case_ids=("second",))
        trainer, _ = self.trainer(batches=[a, b])
        trainer.step(a)
        manifest = trainer.save_checkpoint(self.path / "resume-1")
        original_event = trainer.step(b)
        original_state = copy.deepcopy(self.model.state_dict())
        expected_rng = (random.random(), np.random.random(), torch.rand(5))
        restarted = IdentityModel(self.config)
        restarted.load_state_dict(self.initial)
        second = IdentityTrainer(restarted, trainer.recipe, trainer.admission, first_batch=a)
        second.resume_checkpoint(self.path / "resume-1", expected_manifest_sha256=manifest["manifest_sha256"])
        self.assertEqual((second.steps, second.epoch, second.batch_index), (1, 0, 1))
        actual_event = second.step(b)
        self.assertEqual(original_event, actual_event)
        for key, value in original_state.items():
            torch.testing.assert_close(value, restarted.state_dict()[key], atol=0, rtol=0)
        actual_rng = (random.random(), np.random.random(), torch.rand(5))
        self.assertEqual(expected_rng[:2], actual_rng[:2])
        torch.testing.assert_close(expected_rng[2], actual_rng[2], atol=0, rtol=0)
        self.assertEqual(manifest["training_state"], "OPTIMIZED_UNQUALIFIED")
        self.assertFalse(manifest["acceptance_authority"])

    def test_checkpoint_overwrite_missing_external_pin_and_tampered_payload_are_refused(self):
        trainer, batch = self.trainer()
        trainer.step(batch)
        checkpoint = self.path / "append-only"
        manifest = trainer.save_checkpoint(checkpoint)
        with self.assertRaisesRegex(TeachingError, "new absolute directory"):
            trainer.save_checkpoint(checkpoint)
        with self.assertRaisesRegex(TeachingError, "externally admitted"):
            trainer.resume_checkpoint(checkpoint, expected_manifest_sha256="0" * 64)
        with (checkpoint / "resume.pt").open("ab") as stream:
            stream.write(b"persistent mutation")
        with self.assertRaisesRegex(TeachingError, "payload changed"):
            trainer.resume_checkpoint(checkpoint, expected_manifest_sha256=manifest["manifest_sha256"])

    def test_checkpoint_membership_and_current_source_bindings_are_rechecked(self):
        trainer, batch = self.trainer()
        manifest = trainer.save_checkpoint(self.path / "snapshot")
        extra = self.path / "snapshot" / "extra.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(TeachingError, "membership"):
            trainer.resume_checkpoint(extra.parent, expected_manifest_sha256=manifest["manifest_sha256"])
        extra.unlink()
        Path(trainer.admission.artifacts["reviewed_targets"].path).write_text("{}")
        with self.assertRaisesRegex(TeachingError, "external SHA pin"):
            trainer.resume_checkpoint(extra.parent, expected_manifest_sha256=manifest["manifest_sha256"])

    def test_governed_frame_requires_closed_session_linked_to_exact_frame(self):
        binding, session, kwargs = _production_fixture(self.frame)
        _production(self.frame, binding, session, **kwargs)  # Structural test only; no trainer/gradient grant.
        with self.assertRaisesRegex(TeachingError, "closed production session"):
            _production(self.frame, None, None, **kwargs)
        changed_states = self.frame.sources.states.clone()
        changed_states[:, 0, 0, 0, 0] += 1
        changed_frame = replace(self.frame, sources=replace(self.frame.sources, states=changed_states))
        with self.assertRaisesRegex(TeachingError, "exact frame provenance"):
            _production(changed_frame, binding, session, **kwargs)

    def test_production_fixture_open_session_omitted_member_wrong_code_or_source_is_refused(self):
        binding, session, kwargs = _production_fixture(self.frame)
        for modifications in ({"status": "PUBLIC_MECHANICS_ONLY", "mechanical_fixture_only": True},
                              {"session_source_recheck_passed": False}, {"frame_binding_sha256": ["c" * 64]},
                              {"implementation_sha256": {}}, {"prepared_receipt_file_sha256": "c" * 64},
                              {"prepared_receipt_sha256": "c" * 64}, {"source_acceptance_authority": True}):
            with self.subTest(modifications=modifications), self.assertRaises(TeachingError):
                changed = {**session, **modifications}
                changed.pop("receipt_sha256")
                changed["receipt_sha256"] = sha256(canonical(changed)).hexdigest()
                _production(self.frame, binding, changed, **kwargs)

    def test_production_records_are_part_of_external_reviewed_batch_fingerprint(self):
        batch = _batch(self.model, self.frame)
        binding, session, _ = _production_fixture(self.frame)
        linked = replace(batch, production_binding=binding, production_session=session)
        self.assertNotEqual(linked.fingerprint(), batch.fingerprint())
        changed = replace(linked, production_session={**session, "receipt_sha256": "c" * 64})
        self.assertNotEqual(linked.fingerprint(), changed.fingerprint())

    def test_persistent_source_mutation_during_backward_fails_before_optimizer_update(self):
        trainer, batch = self.trainer()
        before = copy.deepcopy(self.model.state_dict())
        def mutate(gradient):
            Path(trainer.admission.artifacts["source_package"].path).write_bytes(b"mutated public fixture while backward ran")
            return gradient
        handles = [parameter.register_hook(mutate) for parameter in self.model.n2.parameters()]
        with self.assertRaisesRegex(TeachingError, "external SHA pin"):
            trainer.step(batch)
        for handle in handles:
            handle.remove()
        self.assertTrue(trainer.failed)
        self.assertTrue(all(torch.equal(before[key], self.model.state_dict()[key]) for key in before))

    def test_runtime_deterministic_setting_change_is_rejected(self):
        trainer, batch = self.trainer()
        torch.use_deterministic_algorithms(False)
        with self.assertRaisesRegex(TeachingError, "deterministic algorithm"):
            trainer.step(batch)

    def test_governed_training_without_independent_n0_qualification_fails_closed(self):
        with self.assertRaisesRegex(TeachingError, "QUALIFIED_PERSONALITY_N0"):
            self.trainer(reviewed_changes={"source_class": "governed_identity_targets"},
                         auth_changes={"source_class": "governed_identity_targets", "private_gradient_authorized": True})

    def test_returned_evidence_cannot_mutate_checkpoint_authority(self):
        trainer, batch = self.trainer()
        event = trainer.step(batch)
        event["acceptance_authority"] = True
        manifest = trainer.save_checkpoint(self.path / "immutable-evidence")
        self.assertFalse(manifest["events"][0]["acceptance_authority"])
        trainer.events[0]["qualification"] = "QUALIFIED"
        with self.assertRaisesRegex(TeachingError, "cannot claim acceptance"):
            trainer.save_checkpoint(self.path / "forged-evidence")

    def test_admitted_checkpoint_cannot_silently_replace_optimizer_recipe(self):
        trainer, batch = self.trainer()
        trainer.step(batch)
        path = self.path / "optimizer-tamper"
        manifest = trainer.save_checkpoint(path)
        state = torch.load(path / "resume.pt", weights_only=True)
        state["optimizer"]["param_groups"][0]["lr"] = 100.0
        torch.save(state, path / "resume.pt")
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        body["resume_sha256"] = sha256((path / "resume.pt").read_bytes()).hexdigest()
        digest = sha256(canonical(body)).hexdigest()
        (path / "manifest.json").write_bytes(canonical({**body, "manifest_sha256": digest}))
        # Even a caller-supplied external checkpoint digest cannot override the
        # independent exact recipe/code/current authorization admission.
        with self.assertRaisesRegex(TeachingError, "hyperparameters"):
            trainer.resume_checkpoint(path, expected_manifest_sha256=digest)
        self.assertTrue(trainer.failed)


if __name__ == "__main__":
    unittest.main()
