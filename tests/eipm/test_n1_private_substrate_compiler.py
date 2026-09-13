from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alice_personality.n1.compiler import ACTIVE_FILES, ALTERNATIVE_FILE, UNKNOWN_FILE, compile_identity_substrate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compile_identity_substrate_preserves_authority_and_unordered_alternatives(tmp_path) -> None:
    root = tmp_path / "frontier"
    output = tmp_path / "out"
    text = "Direct source evidence."
    e0_id = "e0.1"
    e0 = {
        "unit_id": e0_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest().upper(),
        "provenance_class": "E0",
        "historical_truth_allowed": True,
        "alice_lived_memory": False,
        "semantic_labels": ["value.test"],
        "relationship_contexts": [],
        "use_lanes": ["direct_identity_supervision"],
        "loss_mask": {"direct_identity": True},
        "training_authority": False,
    }
    einf = {
        "curated_id": "einf.1",
        "source_proposal_id": "proposal.einf.1",
        "provenance_class": "E-INF",
        "behavioral_proposal": "Tentative behavior.",
        "scenario": "A scenario.",
        "personality_dimensions": ["value.test"],
        "identity_supporting_E0_unit_ids": [e0_id],
        "context_only_E0_unit_ids": [],
        "excluded_context_E0_unit_ids": [],
        "supporting_raw_EINF_ids": [],
        "historical_Elaina_truth": False,
        "owner_final_review_required": True,
        "training_authority": False,
    }
    direct = {
        "curated_id": "asyn.direct.1",
        "source_proposal_id": "proposal.asyn.1",
        "provenance_class": "A-SYN",
        "behavioral_proposal": "Synthetic runtime behavior.",
        "scenario": "A missing runtime situation.",
        "personality_dimensions": ["value.test"],
        "identity_supporting_E0_unit_ids": [e0_id],
        "context_only_E0_unit_ids": [],
        "excluded_context_E0_unit_ids": [],
        "supporting_raw_EINF_ids": [],
        "historical_Elaina_truth": False,
        "runtime_behavioral_prior_allowed": True,
        "Alice_lived_memory": False,
        "autobiographical_recall_allowed": False,
        "owner_final_review_required": True,
        "training_authority": False,
    }
    base = {
        "curated_base_id": "asyn.base.1",
        "provenance_class": "A-SYN",
        "base_behavioral_policy": "Base synthetic policy.",
        "scenario_family": ["test"],
        "personality_dimensions": ["value.test"],
        "identity_supporting_E0_unit_ids": [e0_id],
        "context_only_E0_unit_ids": [],
        "excluded_context_E0_unit_ids": [],
        "historical_Elaina_truth": False,
        "runtime_behavioral_prior_allowed": True,
        "owner_final_review_required": True,
        "training_authority": False,
    }
    targeted = {
        "curated_id": "asyn.target.1",
        "proposal_id": "proposal.target.1",
        "provenance_class": "A-SYN",
        "behavioral_proposal": "Targeted synthetic policy.",
        "scenario": "Targeted scenario.",
        "personality_dimensions": ["value.test"],
        "identity_supporting_E0_unit_ids": [e0_id],
        "context_only_E0_unit_ids": [],
        "excluded_context_E0_unit_ids": [],
        "supporting_curated_EINF_ids": ["einf.1"],
        "related_curated_ASYN_ids": ["asyn.direct.1"],
        "historical_Elaina_truth": False,
        "runtime_behavioral_prior_allowed": True,
        "owner_final_review_required": True,
        "model_training_authority": False,
    }
    contextual = dict(targeted)
    contextual.update({"curated_id": "asyn.context.1", "proposal_id": "proposal.context.1"})

    rows_by_kind = {
        "E0": [e0],
        "EINF": [einf],
        "ASYN_DIRECT": [direct],
        "ASYN_BASE": [base],
        "ASYN_TARGETED": [targeted],
        "ASYN_CONTEXT": [contextual],
    }
    for kind, relative in ACTIVE_FILES.items():
        _write_jsonl(root / relative, rows_by_kind[kind])

    _write_jsonl(
        root / UNKNOWN_FILE,
        [{"competitor_id": "unknown.1", "training_authority": False, "model_training_authority": False}],
    )
    _write_jsonl(
        root / ALTERNATIVE_FILE,
        [{"competitor_id": "alt.1", "training_authority": False, "hard_negative_authorized": False}],
    )
    (root / "curation_manifest.json").write_text(
        json.dumps(
            {
                "authority_flags": {
                    "E0_mutated": False,
                    "candidate_promotion_to_E0_performed": False,
                    "model_training_performed": False,
                    "owner_final_preweight_review_required": True,
                    "training_authority_granted": False,
                    "weights_created": False,
                },
                "counts": {
                    "curated_EINF": 1,
                    "curated_direct_ASYN_carried": 1,
                    "factorized_base_policies": 1,
                    "targeted_ASYN_added": 1,
                    "context_variants_added": 1,
                    "historical_UNKNOWN_bank": 1,
                    "targeted_gap_queue_remaining": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    receipt = compile_identity_substrate(root, output)
    assert receipt["active_record_count"] == 6
    assert receipt["training_authority_counts"] == {"false": 6}
    assert receipt["owner_final_review_required_counts"] == {"true": 6}
    assert receipt["private_gradient_authorized"] is False
    assert receipt["alternatives_are_unordered_not_negatives"] is True
