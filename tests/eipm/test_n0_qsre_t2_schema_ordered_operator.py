from __future__ import annotations

import torch

from alice_personality.n0.qsre_t2_schema_ordered_operator import (
    QSRET2SchemaOrderedConfig,
    QSRET2SchemaOrderedOperatorEncoder,
)


def _model() -> QSRET2SchemaOrderedOperatorEncoder:
    torch.manual_seed(7)
    cfg = QSRET2SchemaOrderedConfig(
        semantic_dim=32,
        model_dim=64,
        num_hidden_states=5,
        num_attention_heads=4,
        refinement_layers=1,
        num_relations=6,
        num_roles=4,
        num_operations=5,
        num_controls=3,
        max_relation_steps=2,
        executor_context_dim=4,
        dropout=0.0,
    )
    schema = torch.randn(6, 5, 32)
    return QSRET2SchemaOrderedOperatorEncoder(
        cfg,
        relation_schema_hidden_states=schema,
    ).eval()


def test_schema_ordered_shapes_and_report() -> None:
    model = _model()
    query = torch.randn(3, 5, 11, 32)
    mask = torch.ones(3, 11, dtype=torch.bool)
    out = model(
        query_hidden_states=query,
        query_token_mask=mask,
    )

    assert out["relation_logits"].shape == (3, 2, 7)
    assert out["role_logits"].shape == (3, 4)
    assert out["operation_logits"].shape == (3, 5)
    assert out["control_logits"].shape == (3, 3)
    assert out["relation_grounding_attention"].shape == (3, 2, 5, 11)
    assert out["relation_ordered_state"].shape == (3, 2, 64)

    report = model.parameter_report()
    assert report["schema_grounded_relation_readout"] is True
    assert report["learned_relation_class_anchor"] is False
    assert report["ordered_relation_transition"] is True
    assert report["structural_stop_decision"] is True
    assert report["oracle_relation_input"] is False
    assert report["graph_support_input"] is False


def test_relation_schema_is_persistent_frozen_input_buffer() -> None:
    model = _model()
    assert "relation_schema_hidden_states" in dict(model.named_buffers())
    assert "relation_schema_hidden_states" in model.state_dict()
    assert "relation_schema_hidden_states" not in dict(model.named_parameters())


def test_schema_relation_permutation_equivariance() -> None:
    torch.manual_seed(12)
    model = _model()
    query = torch.randn(2, 5, 9, 32)
    mask = torch.ones(2, 9, dtype=torch.bool)

    baseline = model(
        query_hidden_states=query,
        query_token_mask=mask,
    )["relation_logits"].detach()

    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    inverse = torch.argsort(permutation)

    with torch.no_grad():
        original = model.relation_schema_hidden_states.clone()
        model.relation_schema_hidden_states.copy_(original[permutation])

    permuted = model(
        query_hidden_states=query,
        query_token_mask=mask,
    )["relation_logits"].detach()

    # Semantic relation channels move exactly with schema rows. Structural STOP
    # remains the final channel.
    restored = permuted[..., :6][..., inverse]
    assert torch.allclose(
        baseline[..., :6],
        restored,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        baseline[..., 6],
        permuted[..., 6],
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_second_relation_slot_depends_on_previous_relation_distribution() -> None:
    model = _model()
    query = torch.randn(2, 5, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)

    out = model(
        query_hidden_states=query,
        query_token_mask=mask,
    )
    loss = out["relation_logits"][:, 1, 0].sum()
    loss.backward()

    grad = model.relation_transition.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0


def test_masked_tokens_do_not_receive_relation_grounding_attention() -> None:
    model = _model()
    query = torch.randn(2, 5, 8, 32)
    mask = torch.tensor(
        [
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    out = model(
        query_hidden_states=query,
        query_token_mask=mask,
    )

    attn = out["relation_grounding_attention"]
    expanded_mask = mask[:, None, None, :].expand_as(attn)
    assert torch.all(attn.masked_select(~expanded_mask).eq(0))


def test_decode_keeps_structural_stop_fail_closed() -> None:
    model = _model()
    batch = 2
    output = {
        "relation_logits": torch.full((batch, 2, 7), -10.0),
        "role_logits": torch.zeros(batch, 4),
        "operation_logits": torch.zeros(batch, 5),
        "control_logits": torch.zeros(batch, 3),
    }
    output["relation_logits"][0, 0, 6] = 10.0
    output["relation_logits"][0, 1, 4] = 10.0
    output["relation_logits"][1, 0, 4] = 10.0
    output["relation_logits"][1, 1, 6] = 10.0
    output["control_logits"][:, 1] = 10.0

    decoded = model.decode_for_frozen_t1(
        output,
        focus_field_weight=torch.ones(batch, 3),
    )

    assert decoded.relation_sequence_mask.tolist() == [
        [False, False],
        [True, False],
    ]
