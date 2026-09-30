"""CPU-only signed admission and resume binding for an opt-in full fit."""

from __future__ import annotations

from pathlib import Path
import types
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_sha256
from cognitive_kernel.formation_adjudication_v16 import verify_adjudicated_corpus_v16
from cognitive_kernel.formation_learning_v16 import FULL_ROLE_DIMENSIONS
from scripts.mfm import train_v16_formation_specialist as trainer


def arguments(**changes):
    fields = {
        "mode": "train", "probe_only": False, "full_fit": True,
        "admitted_manifest": Path("/fixture/manifest.json"),
        "public_synthetic_curriculum": None,
        "input_sha256": "a" * 64,
        "trust_roster": Path("/steward/trust.json"),
        "trust_roster_sha256": "b" * 64,
        "owner_authorization_ref": None,
        "resume_checkpoint": None,
        "max_source_tokens": 128, "max_target_tokens": 128,
        "specialist_width": 8, "specialist_layers": 1,
        "specialist_heads": 2, "logit_chunk_tokens": 2,
        "epochs": 2, "gradient_accumulation": 1,
        "save_every_steps": 1, "learning_rate": 1e-4,
        "max_cross_attention_pairs": 128, "seed": 7,
    }
    fields.update(changes)
    return types.SimpleNamespace(**fields)


def signed_receipt():
    return {"corpus_manifest_sha256": "a" * 64,
            "externally_pinned_roster_sha256": "b" * 64,
            "rights_and_review_signatures_verified": True,
            "final_payloads_opened": False,
            "qualified_model": False}


class MFMV16SignedFitGateTests(unittest.TestCase):
    def test_missing_or_mixed_full_fit_flags_fail_before_private_open(self):
        for changed, expected in (
                ({"trust_roster": None}, "externally pinned"),
                ({"trust_roster_sha256": None}, "externally pinned"),
                ({"probe_only": True}, "separate routes"),
                ({"public_synthetic_curriculum": Path("/public/seed.jsonl")},
                  "admitted corpus")):
            with self.subTest(changed=changed), \
                    patch.object(trainer, "arguments", return_value=arguments(**changed)), \
                    patch.object(trainer, "_examples",
                                 side_effect=AssertionError("private corpus opened")), \
                    self.assertRaisesRegex(CognitiveKernelContractError, expected):
                trainer.main()

    def test_bad_signature_blocks_optimizer_example_construction(self):
        args = arguments()
        with patch.object(trainer, "admit_formation_corpus", return_value=object()), \
                patch.object(trainer, "verify_adjudicated_corpus_v16",
                             side_effect=CognitiveKernelContractError("invalid review signature")), \
                patch.object(trainer, "admitted_rows_v16",
                             side_effect=AssertionError("unverified target opened")) as rows:
            with self.assertRaisesRegex(CognitiveKernelContractError,
                                        "invalid review signature"):
                trainer._examples(args)
            rows.assert_not_called()
        self.assertFalse(hasattr(args, "signed_review_receipt_sha256"))

    def test_verified_admission_precedes_only_train_development_examples(self):
        args = arguments()
        events = []
        item = types.SimpleNamespace(
            case_id="case-a", adjudications=tuple((name, "present")
                                                   for name in sorted(FULL_ROLE_DIMENSIONS)))
        def verify(*a, **kw):
            events.append("signed-review")
            self.assertEqual(kw["expected_roster_sha256"], args.trust_roster_sha256)
            return signed_receipt()
        def read(admission, *, split):
            events.append(split)
            self.assertEqual(events[0], "signed-review")
            self.assertIn(split, {"train", "development"})
            return iter((item,))
        with patch.object(trainer, "admit_formation_corpus", return_value=object()), \
                patch.object(trainer, "verify_adjudicated_corpus_v16", side_effect=verify), \
                patch.object(trainer, "admitted_rows_v16", side_effect=read), \
                patch.object(trainer, "supervised_output_record_v16"), \
                patch.object(trainer, "model_input_sha256_v16",
                             side_effect=("c" * 64, "d" * 64)):
            train, dev, status = trainer._examples(args)
        self.assertEqual(events, ["signed-review", "train", "development"])
        self.assertEqual((len(train), len(dev)), (1, 1))
        self.assertEqual(status, "admitted-signed-review-final-sealed-unqualified")
        self.assertEqual(args.signed_review_receipt_sha256,
                         canonical_sha256(signed_receipt()))

    def test_real_signed_manifest_keeps_missing_final_files_sealed(self):
        """Use signed fixture with absent FINAL payloads; mock only row decode."""
        try:
            from tests.governance.test_formation_adjudication_v16 import (
                FormationAdjudicationV16Tests, NOW,
            )
        except ModuleNotFoundError:
            self.skipTest("cryptography unavailable in this Python environment")
        signed = FormationAdjudicationV16Tests(
            "test_valid_signatures_bind_complete_review_and_keep_final_sealed")
        signed.setUp()
        try:
            manifest = signed.root / "manifest.json"
            args = arguments(admitted_manifest=manifest,
                             input_sha256=trainer.shared._digest(manifest),
                             trust_roster=signed.root / "trust.json",
                             trust_roster_sha256=signed.roster_sha)
            real_verify = lambda admission, path, roster, **kwargs: (
                verify_adjudicated_corpus_v16(admission, path, roster,
                                              as_of=NOW, **kwargs))
            def decode(admission, *, split):
                self.assertIn(split, {"train", "development"})
                return iter((types.SimpleNamespace(
                    case_id=f"case-{split}", adjudications=tuple(
                        (name, "present") for name in sorted(FULL_ROLE_DIMENSIONS))),))
            with patch.object(trainer, "verify_adjudicated_corpus_v16",
                              side_effect=real_verify), \
                    patch.object(trainer, "admitted_rows_v16", side_effect=decode), \
                    patch.object(trainer, "supervised_output_record_v16"), \
                    patch.object(trainer, "model_input_sha256_v16",
                                 side_effect=("c" * 64, "d" * 64)):
                _, _, status = trainer._examples(args)
            self.assertIn("signed-review", status)
            self.assertFalse((signed.root / "final/source").exists())
            self.assertFalse((signed.root / "final/target").exists())
            self.assertEqual(len(args.signed_review_receipt_sha256), 64)
        finally:
            signed.doCleanups()

    def test_preflight_and_resume_digest_bind_steward_roster(self):
        args = arguments(signed_review_receipt_sha256="c" * 64)
        prepared = {"receipt_sha256": "d" * 64,
                    "files": [{"path": "model.safetensors", "sha256": "e" * 64}]}
        preflight = {"record_sha256": "f" * 64, "transformers_version": "unit"}
        config = types.SimpleNamespace(record=lambda: {"width": 8})
        with patch.object(trainer.shared, "_prepared_kind", return_value="licensed-clone"), \
                patch.object(trainer, "_decoder_sha256", return_value="0" * 64):
            binding = trainer._binding(args, prepared, "signed", 10, 2, "unit")
            run = trainer._run_manifest_v16(args, prepared, preflight,
                                            config, "cpu-test")
            args.trust_roster_sha256 = "1" * 64
            changed_binding = trainer._binding(args, prepared, "signed", 10, 2, "unit")
            changed_run = trainer._run_manifest_v16(args, prepared, preflight,
                                                    config, "cpu-test")
        self.assertEqual(binding["signed_review_receipt_sha256"], "c" * 64)
        self.assertEqual(run["signed_review_receipt_sha256"], "c" * 64)
        self.assertNotEqual(canonical_sha256(binding), canonical_sha256(changed_binding))
        self.assertNotEqual(trainer.shared._record_hash(run),
                            trainer.shared._record_hash(changed_run))
        args.signed_review_receipt_sha256 = None
        with self.assertRaisesRegex(CognitiveKernelContractError, "review binding"):
            trainer._run_manifest_v16(args, prepared, preflight, config, "cpu-test")

    def test_incomplete_positive_coverage_blocks_fit_with_specific_dimensions(self):
        train = [types.SimpleNamespace(adjudications=tuple(
            (name, "negative" if name == "abstention" else "present")
            for name in sorted(FULL_ROLE_DIMENSIONS)))]
        dev = [types.SimpleNamespace(adjudications=tuple(
            (name, "negative" if name == "relationship" else "present")
            for name in sorted(FULL_ROLE_DIMENSIONS)))]
        with self.assertRaises(CognitiveKernelContractError) as captured:
            trainer._require_signed_fit_coverage(train, dev)
        self.assertIn('"train":["abstention"]', str(captured.exception))
        self.assertIn('"development":["relationship"]', str(captured.exception))

    def test_public_synthetic_route_keeps_probe_status_and_no_roster(self):
        args = arguments(mode="data-preflight", full_fit=False,
                         public_synthetic_curriculum=Path("/public/seed.jsonl"),
                         admitted_manifest=None, trust_roster=None,
                         trust_roster_sha256=None,
                         owner_authorization_ref="owner-approved")
        with patch.object(trainer, "_public_examples", return_value=(("train",), ())):
            train, dev, status = trainer._examples(args)
        self.assertEqual((train, dev), (("train",), ()))
        self.assertEqual(status,
                         "owner-attested-public-synthetic-cpu-only-unqualified")
        self.assertFalse(hasattr(args, "signed_review_receipt_sha256"))


if __name__ == "__main__":
    unittest.main()
