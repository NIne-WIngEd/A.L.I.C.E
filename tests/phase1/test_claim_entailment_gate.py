from __future__ import annotations
import sys, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))

from alice_vault.claim_entailment_gate import (
    filter_model_output_by_entailment,
    load_claim_entailment_policy,
)


class FakeNLI:
    """Fake logits follow the active FEVER-NLI policy label order:
    entailment, neutral, contradiction.
    """

    def predict(
        self,
        pairs,
        batch_size=8,
        show_progress_bar=False,
    ):
        out=[]
        for premise,hypothesis in pairs:
            if "U-Net" in premise and "U-Net" in hypothesis:
                # Strong entailment.
                out.append([4.0, 0.2, 0.1])
            else:
                # Strong neutral.
                out.append([0.1, 3.5, 0.2])
        return out


def renderer(claims):
    return "\n".join(
        c["text"]+" "+" ".join(c["citations"])
        for c in claims
    )


class ClaimEntailmentGateTests(unittest.TestCase):
    def policy(self):
        return load_claim_entailment_policy(
            ROOT/"policies"/"claim_entailment_policy.json"
        )

    def test_supported_claim_is_kept_and_neutral_claim_is_dropped(self):
        output={
            "answer_type":"grounded",
            "answer":"old",
            "claims":[
                {
                    "text":"The project used a U-Net.",
                    "claim_type":"fact",
                    "citations":["[S1]"],
                },
                {
                    "text":"The project was published in Nature.",
                    "claim_type":"fact",
                    "citations":["[S1]"],
                },
            ],
            "uncertainty_notes":[],
            "contradiction_notes":[],
        }
        context={
            "evidence":[
                {
                    "citation":"[S1]",
                    "context_text":"The AFM project used a U-Net for segmentation.",
                    "owner_relation":"owner_self_record",
                }
            ]
        }
        filtered,summary=filter_model_output_by_entailment(
            model_output=output,
            context_package=context,
            model=FakeNLI(),
            policy=self.policy(),
            answer_renderer=renderer,
        )
        self.assertEqual(len(filtered["claims"]),1)
        self.assertIn("U-Net",filtered["answer"])
        self.assertNotIn("Nature",filtered["answer"])
        self.assertEqual(summary["dropped_claim_count"],1)

    def test_all_unsupported_claims_fall_back_to_insufficient_evidence(self):
        output={
            "answer_type":"grounded",
            "answer":"old",
            "claims":[
                {
                    "text":"The project was published in Nature.",
                    "claim_type":"fact",
                    "citations":["[S1]"],
                }
            ],
            "uncertainty_notes":[],
            "contradiction_notes":[],
        }
        context={
            "evidence":[
                {
                    "citation":"[S1]",
                    "context_text":"The AFM project used a U-Net.",
                    "owner_relation":"owner_self_record",
                }
            ]
        }
        filtered,summary=filter_model_output_by_entailment(
            model_output=output,
            context_package=context,
            model=FakeNLI(),
            policy=self.policy(),
            answer_renderer=renderer,
        )
        self.assertEqual(
            filtered["answer_type"],
            "insufficient_evidence",
        )
        self.assertEqual(filtered["claims"],[])
        self.assertEqual(summary["kept_claim_count"],0)


if __name__=="__main__":
    unittest.main()
