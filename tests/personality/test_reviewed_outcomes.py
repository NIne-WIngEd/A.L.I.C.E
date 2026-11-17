"""Primary development review consumes actual outputs, never source confidence."""
from dataclasses import replace
from hashlib import sha256
import unittest

from src.alice_personality.identity.codec import canonical
from src.alice_personality.teaching.cases import NamedTarget, TargetFact
from src.alice_personality.teaching.contracts import TeachingError
from src.alice_personality.teaching.lineage import ConstructionLineage
from src.alice_personality.teaching.outcomes import calibration_case_from_review
from tests.personality import test_identity_consumer as consumer_fixture


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = consumer_fixture.ConsumerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.response = self.fixture.build().respond(self.fixture.doc, self.fixture.produced, self.fixture.session)
        self.lineage = ConstructionLineage(("fictional-source-family",), (), ("fictional-new-calibration-episode",))
        self.fact = TargetFact(("opaque-candidate-b",), .7, .1, True, ("opaque-source-id",), ())
        self.target = NamedTarget("owner_fidelity_uncertainty", "independent", (self.fact,))

    def review(self, **changes):
        args = dict(response_receipt=self.response, expected_response_sha256=self.response["receipt_sha256"],
            document=self.fixture.doc, case_id="fictional-primary-outcome", split="CALIBRATION_TRAIN",
            source_class="public_mechanical_fixture", lineage=self.lineage, targets=(self.target,),
            artifact_sha256={name: "a" * 64 for name in ("source_package", "compiled_receipt", "annotation_protocol", "review_record")},
            review_id="fictional-primary-review", annotation_note="Explicit reviewed outcome; source confidence is not fidelity gold.")
        args.update(changes)
        return calibration_case_from_review(**args)

    def test_exact_review_carries_actual_response_model_lineage_and_no_owner_approval(self):
        case = self.review()
        self.assertEqual(case.targets[0].facts[0].value, .7)
        self.assertEqual(case.targets[0].facts[0].uncertainty, .1)
        self.assertEqual(case.outcome_binding["response_receipt_sha256"], self.response["receipt_sha256"])
        self.assertEqual(case.outcome_binding["model_state_sha256"], self.response["model_state_sha256"])
        self.assertEqual(case.outcome_binding["review_scope"], "PRIMARY_DEVELOPMENT_TEXT")
        self.assertFalse(case.outcome_binding["owner_fidelity_approved"])
        self.assertFalse(case.outcome_binding["spoken_fidelity_verified"])
        self.assertEqual(type(case).from_mapping(case.to_mapping()).sha256, case.sha256)

    def test_changed_response_or_document_cannot_be_reviewed_under_old_pin(self):
        changed = dict(self.response, response_text="new response")
        with self.assertRaisesRegex(TeachingError, "response"):
            self.review(response_receipt=changed)
        changed_doc = replace(self.fixture.doc, candidates=(replace(self.fixture.doc.candidates[0], text="different"),
                                                           self.fixture.doc.candidates[1]))
        with self.assertRaisesRegex(TeachingError, "document"):
            self.review(document=changed_doc)

    def test_no_automatic_gold_from_confidence_or_unobserved_response(self):
        with self.assertRaisesRegex(TeachingError, "explicit reviewed"):
            self.review(targets=())
        target = replace(self.target, facts=(replace(self.fact, pointers=("not-an-actual-response",)),))
        with self.assertRaisesRegex(TeachingError, "actual response"):
            self.review(targets=(target,))
        changed = dict(self.response, owner_fidelity_approved=True)
        changed.pop("receipt_sha256")
        changed["receipt_sha256"] = sha256(canonical(changed)).hexdigest()
        with self.assertRaisesRegex(TeachingError, "approval"):
            self.review(response_receipt=changed, expected_response_sha256=changed["receipt_sha256"])
