from __future__ import annotations

import torch

from alice_personality.n0.qsre_production_core import (
    QSREDynamicRelationSchema,
    QSREProductionConfig,
)
from alice_personality.n0.qsre_production_operator_v3 import (
    EVENT_STOP,
    EVENT_UNKNOWN,
    QSREProductionOperatorInducerV3,
)


def cfg() -> QSREProductionConfig:
    return QSREProductionConfig(
        semantic_dim=24,
        model_dim=24,
        num_hidden_states=3,
        num_attention_heads=4,
        operator_refinement_layers=1,
        field_state_dim=24,
        field_metadata_dim=3,
        role_count=4,
        traversal_count=3,
        direction_count=3,
        modifier_count=4,
        control_count=3,
        dropout=0.0,
    )


def schema(relations: int, *, seed: int = 1) -> QSREDynamicRelationSchema:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(relations, 3, 5, 24, generator=g)
    mask = torch.ones(relations, 5, dtype=torch.bool)
    domain = torch.ones(relations, 4, dtype=torch.bool)
    range_mask = torch.ones(relations, 4, dtype=torch.bool)
    symmetric = torch.zeros(relations, dtype=torch.bool)
    return QSREDynamicRelationSchema(
        token_states=tokens,
        token_mask=mask,
        domain_type_mask=domain,
        range_type_mask=range_mask,
        symmetric=symmetric,
    )


def query(batch: int = 2, tokens: int = 7, *, seed: int = 2):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, 3, tokens, 24, generator=g)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return hidden, mask


def encoded_schema_inputs(
    s: QSREDynamicRelationSchema,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Operator-v3 deliberately does not use encoded schema-token states as
    # relation-matching authority, but the public interface still receives
    # them for compatibility with the P1/P3 runtime.
    token = torch.zeros(
        s.token_states.size(0),
        s.token_states.size(1),
        s.token_states.size(2),
        24,
    )
    summary = s.token_states.mean(dim=(1, 2))
    return token, summary


def run(
    model: QSREProductionOperatorInducerV3,
    *,
    q: torch.Tensor,
    qmask: torch.Tensor,
    s: QSREDynamicRelationSchema,
    max_steps: int = 1,
):
    token, summary = encoded_schema_inputs(s)
    return model(
        query_hidden_states=q,
        query_token_mask=qmask,
        schema=s,
        schema_token_state=token,
        schema_relation_state=summary,
        max_steps=max_steps,
    )


def test_v3_relation_hypotheses_remain_continuous_before_binding() -> None:
    torch.manual_seed(9)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    q, qmask = query(batch=2, seed=10)
    s = schema(6, seed=11)
    out = run(
        model,
        q=q,
        qmask=qmask,
        s=s,
        max_steps=3,
    )["operator"].relation_distribution
    assert torch.isfinite(out).all()
    assert torch.all(out > 0)
    assert torch.allclose(
        out.sum(dim=-1),
        torch.ones_like(out.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_v3_schema_permutation_equivariance() -> None:
    torch.manual_seed(10)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    q, qmask = query(batch=2, seed=11)
    base_schema = schema(5, seed=12)
    base = run(
        model,
        q=q,
        qmask=qmask,
        s=base_schema,
        max_steps=2,
    )["operator"].relation_distribution.detach()

    perm = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.argsort(perm)
    moved_schema = QSREDynamicRelationSchema(
        token_states=base_schema.token_states[perm],
        token_mask=base_schema.token_mask[perm],
        domain_type_mask=base_schema.domain_type_mask[perm],
        range_type_mask=base_schema.range_type_mask[perm],
        symmetric=base_schema.symmetric[perm],
    )
    moved = run(
        model,
        q=q,
        qmask=qmask,
        s=moved_schema,
        max_steps=2,
    )["operator"].relation_distribution.detach()
    assert torch.allclose(
        base,
        moved[..., inverse],
        atol=1e-5,
        rtol=1e-5,
    )


def test_v3_p1_schema_relation_state_is_not_relation_match_authority() -> None:
    torch.manual_seed(41)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    q, qmask = query(batch=1, seed=42)
    s = schema(4, seed=43)
    token, summary = encoded_schema_inputs(s)
    a = model(
        query_hidden_states=q,
        query_token_mask=qmask,
        schema=s,
        schema_token_state=token,
        schema_relation_state=summary,
        max_steps=2,
    )["operator"]
    b = model(
        query_hidden_states=q,
        query_token_mask=qmask,
        schema=s,
        schema_token_state=token,
        schema_relation_state=torch.randn_like(summary) * 1000.0,
        max_steps=2,
    )["operator"]
    assert torch.allclose(
        a.relation_distribution,
        b.relation_distribution,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a.relation_step_mass,
        b.relation_step_mass,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a.continuous_state,
        b.continuous_state,
        atol=1e-6,
        rtol=1e-6,
    )


def test_v3_factor_slots_do_not_depend_on_relation_cardinality() -> None:
    torch.manual_seed(13)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    q, qmask = query(batch=1, seed=14)
    small = schema(3, seed=15)
    extra = schema(2, seed=16)
    large = QSREDynamicRelationSchema(
        token_states=torch.cat([small.token_states, extra.token_states], dim=0),
        token_mask=torch.cat([small.token_mask, extra.token_mask], dim=0),
        domain_type_mask=torch.cat(
            [small.domain_type_mask, extra.domain_type_mask],
            dim=0,
        ),
        range_type_mask=torch.cat(
            [small.range_type_mask, extra.range_type_mask],
            dim=0,
        ),
        symmetric=torch.cat([small.symmetric, extra.symmetric], dim=0),
    )
    a = run(model, q=q, qmask=qmask, s=small)["operator"]
    b = run(model, q=q, qmask=qmask, s=large)["operator"]
    for name in (
        "role_distribution",
        "traversal_distribution",
        "direction_distribution",
        "modifier_weight",
        "control_distribution",
        "applicability",
    ):
        assert torch.allclose(
            getattr(a, name),
            getattr(b, name),
            atol=1e-6,
            rtol=1e-6,
        )


def test_v3_relation_scale_cannot_suppress_stop_unknown_event() -> None:
    torch.manual_seed(17)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    q, qmask = query(batch=2, seed=18)
    s = schema(4, seed=19)
    with torch.no_grad():
        model.relation_logit_scale.fill_(-2.0)
    low = run(model, q=q, qmask=qmask, s=s)["event_distribution"]
    with torch.no_grad():
        model.relation_logit_scale.fill_(6.0)
    high = run(model, q=q, qmask=qmask, s=s)["event_distribution"]
    assert torch.allclose(low, high, atol=1e-6, rtol=1e-6)


def test_v3_unknown_and_stop_are_distinct_events() -> None:
    torch.manual_seed(20)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    q, qmask = query(batch=1, seed=21)
    s = schema(3, seed=22)

    with torch.no_grad():
        model.continue_head.weight.zero_()
        model.stop_head.weight.zero_()
        model.unknown_head.weight.zero_()
        model.continue_head.bias.fill_(-30.0)
        model.stop_head.bias.fill_(30.0)
        model.unknown_head.bias.fill_(-30.0)
        model.event_match_projection.weight.zero_()
    stopped = run(model, q=q, qmask=qmask, s=s)["event_distribution"]

    with torch.no_grad():
        model.stop_head.bias.fill_(-30.0)
        model.unknown_head.bias.fill_(30.0)
    unknown = run(model, q=q, qmask=qmask, s=s)["event_distribution"]

    assert float(stopped[0, 0, EVENT_STOP]) > 0.999
    assert float(stopped[0, 0, EVENT_UNKNOWN]) < 1.0e-6
    assert float(unknown[0, 0, EVENT_UNKNOWN]) > 0.999
    assert float(unknown[0, 0, EVENT_STOP]) < 1.0e-6


def test_v3_schema_identity_logits_are_permutation_equivariant() -> None:
    torch.manual_seed(23)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    s = schema(5, seed=24)
    base = model.schema_identity_logits(s).detach()

    perm = torch.tensor([4, 1, 3, 0, 2])
    inverse = torch.argsort(perm)
    moved_schema = QSREDynamicRelationSchema(
        token_states=s.token_states[perm],
        token_mask=s.token_mask[perm],
        domain_type_mask=s.domain_type_mask[perm],
        range_type_mask=s.range_type_mask[perm],
        symmetric=s.symmetric[perm],
    )
    moved = model.schema_identity_logits(moved_schema).detach()
    restored = moved[inverse][:, inverse]
    assert torch.allclose(base, restored, atol=1e-5, rtol=1e-5)


def test_v3_parameter_report_has_no_schema_or_hop_parameter_axis() -> None:
    report = QSREProductionOperatorInducerV3(cfg()).parameter_report()
    assert report["relation_count_dependent_parameters"] == 0
    assert report["hop_count_dependent_parameters"] == 0
    assert report["runtime_dynamic_relation_schema"] is True
    assert report["continuous_relation_hypotheses"] is True
    assert report["relation_sparsity_before_structural_binding"] is False
    assert report["exact_sparsity_owned_by_binder"] is True
    assert report["shared_query_schema_metric"] is True
    assert report["relation_selection_decoupled_from_stop_unknown"] is True
    assert report["p1_schema_relation_state_is_interface_only_for_operator"] is True
    assert report["factor_specific_query_slots"] is True
    assert report["relation_count_ceiling"] is None
    assert report["runtime_step_count_ceiling"] is None
