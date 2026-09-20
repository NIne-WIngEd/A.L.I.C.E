from __future__ import annotations

import torch

from alice_personality.n0.qsre_n0_bridge import QSREN0EvidenceBridge


def test_bridge_preserves_base_evidence_and_appends_one_slot() -> None:
    bridge = QSREN0EvidenceBridge(semantic_dim=8)
    base = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    fields = torch.randn(2, 4, 8)
    field_mask = torch.ones(2, 4, dtype=torch.bool)
    probability = torch.tensor(
        [[0.0, 0.8, 0.0, 0.0], [0.2, 0.0, 0.3, 0.0]]
    )
    out = bridge(
        base_evidence_tokens=base,
        base_evidence_mask=mask,
        field_semantic_state=fields,
        field_valid_mask=field_mask,
        relational_probability=probability,
    )
    assert out.evidence_tokens.shape == (2, 4, 8)
    assert out.evidence_mask.shape == (2, 4)
    assert torch.equal(out.evidence_tokens[:, :3], base)
    assert torch.equal(out.evidence_mask[:, :3], mask)


def test_bridge_nonrelational_pass_through_is_exact_and_masked() -> None:
    bridge = QSREN0EvidenceBridge(semantic_dim=6)
    base = torch.randn(1, 2, 6)
    mask = torch.tensor([[True, True]])
    fields = torch.randn(1, 5, 6)
    out = bridge(
        base_evidence_tokens=base,
        base_evidence_mask=mask,
        field_semantic_state=fields,
        field_valid_mask=torch.ones(1, 5, dtype=torch.bool),
        relational_probability=torch.zeros(1, 5),
    )
    assert torch.equal(out.evidence_tokens[:, :2], base)
    assert torch.equal(out.evidence_mask[:, :2], mask)
    assert out.evidence_mask[0, 2].item() is False
    assert torch.equal(
        out.relational_semantic_token,
        torch.zeros_like(out.relational_semantic_token),
    )
    assert float(out.relational_execution_confidence[0]) == 0.0


def test_bridge_plural_semantics_are_weighted_without_hard_winner() -> None:
    bridge = QSREN0EvidenceBridge(semantic_dim=3)
    base = torch.zeros(1, 1, 3)
    fields = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    probability = torch.tensor([[0.25, 0.75, 0.0]])
    out = bridge(
        base_evidence_tokens=base,
        base_evidence_mask=torch.ones(1, 1, dtype=torch.bool),
        field_semantic_state=fields,
        field_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        relational_probability=probability,
    )
    expected = torch.tensor([[0.25, 0.75, 0.0]])
    assert torch.allclose(out.relational_semantic_token, expected)
    assert torch.allclose(out.relational_field_distribution, probability)
    assert torch.allclose(out.relational_execution_confidence, torch.ones(1))


def test_bridge_respects_field_validity_and_keeps_calibrated_mass() -> None:
    bridge = QSREN0EvidenceBridge(semantic_dim=4)
    fields = torch.randn(1, 3, 4)
    probability = torch.tensor([[0.2, 0.4, 0.3]])
    out = bridge(
        base_evidence_tokens=torch.randn(1, 2, 4),
        base_evidence_mask=torch.ones(1, 2, dtype=torch.bool),
        field_semantic_state=fields,
        field_valid_mask=torch.tensor([[True, False, True]]),
        relational_probability=probability,
    )
    assert torch.allclose(
        out.relational_execution_confidence,
        torch.tensor([0.5]),
    )
    assert torch.allclose(
        out.relational_field_distribution,
        torch.tensor([[0.4, 0.0, 0.6]]),
    )


def test_bridge_has_no_capacity_ceiling_or_adapter_parameters() -> None:
    report = QSREN0EvidenceBridge(semantic_dim=8).parameter_report()
    assert report["total_parameters"] == 0
    assert report["learned_adapter_bottleneck"] is False
    assert report["field_count_ceiling"] is None
    assert report["support_count_ceiling"] is None
    assert report["candidate_count_ceiling"] is None


def test_bridge_preserves_low_but_nonzero_calibrated_relation_without_second_router() -> None:
    bridge = QSREN0EvidenceBridge(semantic_dim=3)
    fields = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]
    )
    probability = torch.tensor([[0.004, 0.006]])
    out = bridge(
        base_evidence_tokens=torch.zeros(1, 1, 3),
        base_evidence_mask=torch.ones(1, 1, dtype=torch.bool),
        field_semantic_state=fields,
        field_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        relational_probability=probability,
    )
    assert out.relational_token_valid.item() is True
    assert torch.allclose(
        out.relational_execution_confidence,
        torch.tensor([0.01]),
        atol=1e-7,
    )
    normalized = torch.tensor([[0.4, 0.6]])
    expected_semantic = torch.einsum("bf,bfd->bd", normalized, fields)
    assert torch.allclose(
        out.relational_semantic_token,
        expected_semantic * 0.01,
        atol=1e-7,
    )
    report = bridge.parameter_report()
    assert report["hard_relational_confidence_gate"] is False
