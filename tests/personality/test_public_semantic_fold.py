"""PUBLIC mechanics: holdout descriptions cannot enter either TRAIN loss side."""
import importlib
from hashlib import sha256
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tests.personality import test_public_feature_handoff as fixtures
from src.alice_personality.gemma_n0 import public_semantic_experiment as producer


class PublicSemanticFoldTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module("src.alice_personality.gemma_n0.public_semantic_fold")
        except ModuleNotFoundError:self.fail("the predeclared TRAIN-family repair consumer is missing")

    def fixture(self):
        plans,rows=[],[]
        banks=[object() for _ in range(8)]
        for i in range(8):
            texts=["public description "+str(j) for j in range(8)]
            plans.append({"id_metadata":"case"+str(i),"family_metadata":"family"+str(i),"split":"train",
                          "descriptions":texts,"target_index_metadata":i})
            rows.append({"id":"case"+str(i),"family":"family"+str(i),"split":"train",
                         "source":object(),"candidates":banks,"target":i})
        plans.append({"id_metadata":"dev","family_metadata":"dev-family","split":"dev",
                      "descriptions":["DEV ONLY"],"target_index_metadata":0})
        rows.append({"id":"dev","family":"dev-family","split":"dev","source":object(),"candidates":[object()],"target":0})
        return SimpleNamespace(plan={"examples":plans},examples=rows)

    def test_no_heldout_or_dev_description_enters_positive_or_negative_training(self):
        fold=self.module().build_fold(self.fixture())
        self.assertEqual(len(fold["train"]),6)
        self.assertEqual(len(fold["heldout"]),2)
        self.assertEqual(len(fold["descriptions"]),8)
        self.assertEqual({row["family"] for row in fold["train"]},set(fold["protocol"]["train_families_metadata"]))
        training_ids=set(fold["protocol"]["train_candidate_input_sha256"])
        self.assertTrue(training_ids.isdisjoint(fold["protocol"]["heldout_candidate_input_sha256"]))
        for row in fold["train"]:
            self.assertEqual(row["candidate_input_sha256"],fold["protocol"]["train_candidate_input_sha256"])
            self.assertEqual(len(row["candidates"]),6)
            self.assertEqual(row["candidate_input_sha256"][row["target"]],row["target_input_sha256"])
        for row in fold["evaluation"]:
            self.assertEqual(len(row["candidates"]),8)
            self.assertEqual(row["candidate_input_sha256"][row["target"]],row["target_input_sha256"])

    def test_fold_assignment_and_pointers_are_independent_of_original_candidate_order(self):
        fixture=self.fixture();a=self.module().build_fold(fixture)
        for meta,row in zip(fixture.plan["examples"],fixture.examples):
            meta["descriptions"].reverse();row["candidates"]=list(reversed(row["candidates"]))
            meta["target_index_metadata"]=len(meta["descriptions"])-1-meta["target_index_metadata"]
            row["target"]=len(row["candidates"])-1-row["target"]
        b=self.module().build_fold(fixture)
        self.assertEqual(a["protocol"],b["protocol"])
        self.assertEqual([r["target_input_sha256"] for r in a["evaluation"]],[r["target_input_sha256"] for r in b["evaluation"]])

    def test_inconsistent_family_description_is_rejected(self):
        fixture=self.fixture()
        fixture.plan["examples"][1]["descriptions"][0]="different declaration"
        with self.assertRaises(ValueError):self.module().build_fold(fixture)

    def test_closed_fixture_round_trip_three_variants_and_no_authority(self):
        fixture=fixtures.PublicFeatureHandoffTests("test_fixture_scope_external_pins_and_create_only")
        self.addCleanup(fixture.doCleanups);fixture.setUp();fixture.export()
        before_rng=torch.get_rng_state().clone();before_threads=torch.get_num_threads()
        module=self.module();original_score=module.diagnostic._score
        def inference_score(*args,**kwargs):
            self.assertFalse(torch.is_grad_enabled(),"receipt scoring must not build training graphs")
            return original_score(*args,**kwargs)
        with patch.object(producer,"_load_plan",side_effect=AssertionError("source reopening forbidden")), \
                patch.object(producer,"_runtime",side_effect=AssertionError("publisher loading forbidden")), \
                patch.object(module.diagnostic,"_score",side_effect=inference_score):
            receipt=module.fit_public_semantic_fold(fixture.package,
                expected_manifest_sha256=fixture.pins["manifest_sha256"],
                expected_closure_sha256=fixture.pins["closure_sha256"],output_directory=fixture.root/"fold",
                allow_mechanics_only=True)
        fixture.provider_factory.assert_not_called()
        self.assertTrue(torch.equal(torch.get_rng_state(),before_rng));self.assertEqual(torch.get_num_threads(),before_threads)
        self.assertEqual(receipt["status"],"PUBLIC_MECHANICS_ONLY")
        self.assertEqual(set(receipt["exports"]),{"fresh_original","anchored_full","anchored_candidate_only"})
        self.assertEqual(len(receipt["per_example"]),6)
        self.assertEqual(receipt["optimizer_updates_per_variant"],2)
        for helper in (module.producer,module.diagnostic,module.partial):
            path=Path(helper.__file__)
            self.assertEqual(receipt["binding"]["code"][path.name],sha256(path.read_bytes()).hexdigest())
        for entry in receipt["training"].values():self.assertTrue(entry["final_optimizer_update_replayed_exact"])
        for key in ("n0_approved","personality_qualified","repair_selected","gemma_neutrality_established",
                    "private_identity_data","publisher_checkpoint_loaded","publisher_forward_performed","final_payload_opened","dev_used_for_fit_or_scoring"):
            self.assertIs(receipt[key],False)
        with self.assertRaises(ValueError):
            self.module().fit_public_semantic_fold(fixture.package,expected_manifest_sha256=fixture.pins["manifest_sha256"],
                expected_closure_sha256=fixture.pins["closure_sha256"],output_directory=fixture.root/"forbidden")

    def test_changed_artificial_source_refuses_result_receipt(self):
        fixture=fixtures.PublicFeatureHandoffTests("test_fixture_scope_external_pins_and_create_only")
        self.addCleanup(fixture.doCleanups);fixture.setUp();fixture.export()
        module=self.module();original_source=module.partial.artificial_source;original_new=module._new
        captured=[]
        def capture(*args,**kwargs):
            bank=original_source(*args,**kwargs);captured.append(bank);return bank
        def mutate_after_plan(*args,**kwargs):
            captured[0].layers[0,0,0]=1
            return original_new(*args,**kwargs)
        with patch.object(module.partial,"artificial_source",side_effect=capture),patch.object(module,"_new",side_effect=mutate_after_plan):
            with self.assertRaisesRegex(ValueError,"artificial source changed"):
                module.fit_public_semantic_fold(fixture.package,expected_manifest_sha256=fixture.pins["manifest_sha256"],
                    expected_closure_sha256=fixture.pins["closure_sha256"],output_directory=fixture.root/"changed-null",allow_mechanics_only=True)
        self.assertFalse((fixture.root/"changed-null"/"fold-fit.json").exists())


if __name__ == "__main__":unittest.main()
