import copy
import unittest

from scripts.eipm.gemma_n0 import magnolia_saved_path_readback_20261020 as api


class SavedPathTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for family in ("one", "two"):
            for kind in sorted(api.KINDS):
                changed = kind == "decisive_condition"
                decision = {
                    "candidate_ids": [["a", "b"]], "concept_ids": [["c"]], "candidate_mask": [[True, True]],
                    "conditional_concepts": {**{name: [[.8 if changed else .2]] for name in
                        ("concept_applicability", "concept_uncertainty", "concept_source_support", "concept_source_disconfirmation")},
                        "concept_source_mask": [[[True]]]},
                    "native_latent": {"candidates": [[[.3, .5], [.2, .4]]], "concepts": [[[1., 2.]]],
                                      "concept_mask": [[True]], "views": {"identity": [[[1., 2.]]]},
                                      "view_masks": {"identity": [[True]]}},
                    **{name: [[.2, .8]] for name in ("co_valid_probabilities", "contraindications", "preferences")}}
                rows.append({"document_id": family+kind, "parent_document_id": family,
                             "comparison_kind": kind, "response": {"decision": decision}})
        return rows

    def test_locates_changed_n1_and_unchanged_n2_without_claiming_cause(self):
        result = api.summarize(self.fixture())
        self.assertAlmostEqual(result["decisive_condition"]["n1:concept_applicability"]["max_absolute_delta"], .6)
        self.assertEqual(result["decisive_condition"]["n2:candidates"]["max_absolute_delta"], 0.)
        self.assertEqual(result["paraphrase"]["n1:concept_applicability"]["max_absolute_delta"], 0.)
        self.assertEqual(result["decisive_condition"]["n1:concept_applicability"]["paired_families"], 2)

    def test_incomplete_or_different_semantic_coordinates_refused(self):
        rows = self.fixture()
        with self.assertRaises(ValueError):
            api.summarize(rows[:-1])
        other = copy.deepcopy(rows)
        other[0]["response"]["decision"]["candidate_ids"] = [["b", "a"]]
        with self.assertRaises(ValueError):
            api.summarize(other)
        for left, right in (([1.], [1., 2.]), ([True], [1.]), ([float("nan")], [0.])):
            with self.subTest(left=left), self.assertRaises(ValueError):
                api.difference(left, right)

    def test_known_coordinate_distance(self):
        value = api.difference([[3., 4.]], [[6., 8.]])
        self.assertEqual(value["coordinates"], 2)
        self.assertEqual(value["max_absolute_delta"], 4.)
        self.assertEqual(value["relative_rms_delta"], 1.)

    def test_guard_refuses_before_any_path_operation(self):
        class Refused(Exception):
            pass
        class Namespace:
            def _p2_guard(self):
                raise Refused()
        with self.assertRaises(Refused):
            api.run(Namespace())
