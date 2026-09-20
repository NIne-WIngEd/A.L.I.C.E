from __future__ import annotations

import torch

from alice_personality.n0.qsre_t2_operator import (
    QSRET2OperatorConfig,
    QSRET2OperatorEncoder,
)


def build_model() -> QSRET2OperatorEncoder:
    torch.manual_seed(7)
    return QSRET2OperatorEncoder(
        QSRET2OperatorConfig(
            semantic_dim=32,
            model_dim=64,
            num_hidden_states=5,
            num_attention_heads=4,
            refinement_layers=2,
            num_relations=6,
            num_roles=4,
            num_operations=5,
            num_controls=3,
            max_relation_steps=2,
            executor_context_dim=4,
            dropout=0.0,
        )
    ).eval()


def test_shapes_and_diagnostics() -> None:
    model = build_model()
    hidden = torch.randn(3, 5, 11, 32)
    mask = torch.ones(3, 11, dtype=torch.bool)
    mask[1, -3:] = False

    out = model(
        query_hidden_states=hidden,
        query_token_mask=mask,
    )

    assert out["relation_logits"].shape == (3, 2, 7)
    assert out["role_logits"].shape == (3, 4)
    assert out["operation_logits"].shape == (3, 5)
    assert out["control_logits"].shape == (3, 3)
    assert out["continuous_state"].shape == (3, 64)
    assert out["layer_attention"].shape == (3, 6, 5)
    assert out["token_layer_attention"].shape == (3, 6, 5, 11)
    assert torch.isfinite(out["uncertainty"]).all()


def test_masked_tokens_receive_zero_attention() -> None:
    model = build_model()
    hidden = torch.randn(2, 5, 9, 32)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 6:] = False

    out = model(
        query_hidden_states=hidden,
        query_token_mask=mask,
    )
    assert torch.equal(
        out["token_layer_attention"][0, :, :, 6:],
        torch.zeros_like(out["token_layer_attention"][0, :, :, 6:]),
    )


def test_decode_uses_external_oracle_focus_only() -> None:
    model = build_model()
    hidden = torch.randn(2, 5, 7, 32)
    mask = torch.ones(2, 7, dtype=torch.bool)
    out = model(query_hidden_states=hidden, query_token_mask=mask)

    # Force a deterministic decoded operator without supplying graph/support
    # information to the encoder itself.
    out["relation_logits"].fill_(-10.0)
    out["relation_logits"][:, 0, 2] = 10.0
    out["relation_logits"][:, 1, model.none_relation_id] = 10.0
    out["role_logits"].fill_(-10.0)
    out["role_logits"][:, 1] = 10.0
    out["operation_logits"].fill_(-10.0)
    out["operation_logits"][:, 0] = 10.0
    out["control_logits"].fill_(-10.0)
    out["control_logits"][:, 1] = 10.0

    focus = torch.zeros(2, 6)
    focus[:, 3] = 1.0
    operator = model.decode_for_frozen_t1(out, focus_field_weight=focus)

    assert operator.relation_sequence_mask.tolist() == [
        [True, False],
        [True, False],
    ]
    assert operator.relation_sequence_id[:, 0].tolist() == [2, 2]
    assert operator.role_id.tolist() == [1, 1]
    assert operator.operation_id.tolist() == [0, 0]
    assert torch.equal(operator.focus_field_weight, focus)
    assert torch.equal(operator.context, torch.zeros(2, 4))
    assert torch.allclose(operator.applicability, torch.full((2,), 0.9))


def test_nonrelational_control_closes_relation_sequence() -> None:
    model = build_model()
    hidden = torch.randn(1, 5, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = model(query_hidden_states=hidden, query_token_mask=mask)
    out["relation_logits"].fill_(-10.0)
    out["relation_logits"][:, :, 1] = 10.0
    out["control_logits"].fill_(-10.0)
    out["control_logits"][:, 2] = 10.0

    operator = model.decode_for_frozen_t1(
        out,
        focus_field_weight=torch.zeros(1, 4),
    )
    assert not bool(operator.relation_sequence_mask.any())
    assert torch.allclose(operator.applicability, torch.tensor([0.5]))


def test_parameter_report_preserves_causal_boundary() -> None:
    report = build_model().parameter_report()
    assert report["semantic_backbone_parameters"] == 0
    assert report["oracle_relation_input"] is False
    assert report["oracle_role_input"] is False
    assert report["graph_support_input"] is False
    assert report["multi_layer_token_input"] is True
    assert report["schema_anchor_readout"] is True
    assert report["hard_layer_mask"] is False
    assert report["max_relation_steps_is_product_ceiling"] is False
