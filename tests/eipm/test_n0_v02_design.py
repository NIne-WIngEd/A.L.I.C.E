from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.v02_objectives import (
    multi_positive_contrastive_loss,
    principle_alignment_examples,
    principle_alignment_loss,
    symmetric_contrastive_loss,
    validate_teacher_row_for_v02,
    weighted_objective_sum,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/eipm/n0/alice_n0_semantic_v0.2.json"
MIXTURE = ROOT / "configs/eipm/n0/public_corpus_v0.2.plan.json"
BENCHMARK = ROOT / "evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl"


def test_v02_architecture_is_native_compute_efficient_and_30pct_masked() -> None:
    config = load_n0_config(CONFIG)
    assert config.model_id == "alice-n0-semantic-v0.2"
    assert config.hidden_size == 640
    assert config.num_hidden_layers == 16
    assert config.num_attention_heads == 10
    assert config.hidden_size // config.num_attention_heads == 64
    assert config.intermediate_size == 2560
    assert config.mlm_probability == pytest.approx(0.30)
    assert config.planned_parameter_count == 136_594_435

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert raw["architecture"]["third_party_weight_initialization"] is False
    assert raw["architecture"]["exact_parameter_count"] == 136_594_435
    assert raw["governance"]["private_identity_gradient"] is False
    assert raw["governance"]["v01_pathfinder_weights_must_not_initialize_v02"] is True


def test_v02_objective_weights_are_explicit_and_normalized() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    objectives = raw["training"]["objectives"]
    active = [entry for entry in objectives.values() if entry.get("enabled")]
    assert sum(float(entry["base_weight"]) for entry in active) == pytest.approx(1.0)
    assert objectives["principle_rationale_alignment"]["rationale_text_must_supply_gradient"] is True
    assert objectives["replaced_token_detection"]["enabled"] is False


def test_v02_mixture_is_balanced_and_not_activated_without_rights_schema() -> None:
    raw = json.loads(MIXTURE.read_text(encoding="utf-8"))
    assert raw["status"] == "planned_not_activated"
    assert raw["global_requirements"]["row_level_license_required"] is True
    assert raw["global_requirements"]["row_level_provenance_required"] is True
    assert sum(float(group["target_share"]) for group in raw["mixture"]) == pytest.approx(1.0)
    source_shares = [
        float(source["target_share"])
        for group in raw["mixture"]
        for source in group["sources"]
    ]
    assert sum(source_shares) == pytest.approx(1.0)
    assert max(source_shares) <= 0.125
    assert raw["tokenizer_sampling"]["deterministic_source_balancing_required"] is True
    assert raw["storage_and_tranche_policy"]["do_not_require_full_corpus_residency_on_magnolia"] is True

    by_category = {
        group["category"]: {source["repo_id"] for source in group["sources"]}
        for group in raw["mixture"]
    }
    assert "common-pile/stackv2_edu_filtered" not in by_category["educational_qa_explanation"]
    assert "common-pile/stackv2_edu_filtered" in by_category["software_code_discussion"]
    assert {
        "common-pile/libretexts_filtered",
        "common-pile/pressbooks_filtered",
        "common-pile/oercommons_filtered",
    }.issubset(by_category["educational_qa_explanation"])


def test_fixed_v02_suite_is_eval_only_and_covers_all_43_competencies() -> None:
    rows = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 43
    assert len({row["competency"] for row in rows}) == 43
    assert len({row["id"] for row in rows}) == 43
    assert all(row["eval_only"] is True for row in rows)
    assert all(row["training_authorized"] is False for row in rows)
    assert all(row["prompt"].strip() != row["prompt_paraphrase"].strip() for row in rows)


def test_rationale_becomes_gradient_bearing_supervision() -> None:
    row = {
        "id": "x",
        "competency": "EPI-04",
        "principle_tag": "preserve_uncertainty",
        "prompt": "Weak evidence supports two readings. What should happen?",
        "candidates": ["preserve uncertainty", "claim certainty"],
        "preferred_indices": [0],
        "rationale": "Confidence should track unresolved evidence.",
    }
    validate_teacher_row_for_v02(row)
    examples = principle_alignment_examples(row)
    assert [example["label"] for example in examples] == [1.0, 0.0]
    assert all(example["rationale"] == row["rationale"] for example in examples)
    assert all(example["principle_tag"] == "preserve_uncertainty" for example in examples)

    logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])
    assert principle_alignment_loss(logits, labels).item() < 0.2


def test_contrastive_and_weighted_objectives_are_finite() -> None:
    semantic = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rationale = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    contrastive = symmetric_contrastive_loss(semantic, rationale, temperature=0.1)
    assert torch.isfinite(contrastive)

    combined = weighted_objective_sum(
        {"span_mlm": torch.tensor(2.0), "semantic_contrastive": torch.tensor(1.0)},
        {"span_mlm": 0.7, "semantic_contrastive": 0.1},
    )
    assert combined.item() == pytest.approx(1.875)


def test_multi_positive_contrastive_does_not_make_same_principle_false_negatives() -> None:
    semantic = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rationale = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    loss = multi_positive_contrastive_loss(
        semantic,
        rationale,
        ["preserve_uncertainty", "preserve_uncertainty", "relation_direction"],
        temperature=0.1,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
