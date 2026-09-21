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


def relation_schema(relations: int, *, seed: int = 1) -> QSREDynamicRelationSchema:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(relations, 3, 5, 24, generator=g)
    return QSREDynamicRelationSchema(
        token_states=tokens,
        token_mask=torch.ones(relations, 5, dtype=torch.bool),
        domain_type_mask=torch.ones(relations, 4, dtype=torch.bool),
        range_type_mask=torch.ones(relations, 4, dtype=torch.bool),
        symmetric=torch.zeros(relations, dtype=torch.bool),
    )


def factor_cache(*, seed: int = 10) -> dict:
    g = torch.Generator().manual_seed(seed)

    def categorical(count: int) -> dict:
        return {
            "keys": [f"k{i}" for i in range(count)],
            "token_states": torch.randn(count, 3, 4, 24, generator=g),
            "token_mask": torch.ones(count, 4, dtype=torch.bool),
        }

    return {
        "schema": "alice.eipm.n0.qsre-closure-factor-schema-cache.v1",
        "private_identity_data": False,
        "role": categorical(4),
        "traversal": categorical(3),
        "direction": categorical(3),
        "control": categorical(3),
        "modifiers": {
            "keys": [f"m{i}" for i in range(4)],
            "token_states": torch.randn(4, 2, 3, 4, 24, generator=g),
            "token_mask": torch.ones(4, 2, 4, dtype=torch.bool),
        },
    }


def query(batch: int = 2, tokens: int = 7, *, seed: int = 20):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, 3, tokens, 24, generator=g)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return hidden, mask


def make_model(seed: int = 30) -> QSREProductionOperatorInducerV3:
    torch.manual_seed(seed)
    model = QSREProductionOperatorInducerV3(cfg()).eval()
    model.configure_factor_schema_cache(factor_cache(seed=seed + 1))
    return model


def run(
    model: QSREProductionOperatorInducerV3,
    *,
    q: torch.Tensor,
    qmask: torch.Tensor,
    schema: QSREDynamicRelationSchema,
    max_steps: int = 2,
    schema_relation_state: torch.Tensor | None = None,
):
    if schema_relation_state is None:
        schema_relation_state = schema.token_states.mean(dim=(1, 2))
    return model(
        query_hidden_states=q,
        query_token_mask=qmask,
        schema=schema,
        schema_token_state=torch.zeros(
            schema.token_states.size(0),
            3,
            5,
            24,
        ),
        schema_relation_state=schema_relation_state,
        max_steps=max_steps,
    )


def test_closure_operator_has_no_fixed_factor_class_heads() -> None:
    model = make_model()
    for name in (
        "role_head",
        "traversal_head",
        "direction_head",
        "modifier_head",
        "control_head",
    ):
        assert not hasattr(model, name)
    report = model.parameter_report()
    assert report["fixed_factor_class_head_parameters"] == 0
    assert report["semantic_factor_schemas"] is True
    assert report["ordered_query_evidence_coverage"] is True
    assert report["relation_count_ceiling"] is None
    assert report["runtime_step_count_ceiling"] is None


def test_pretrained_schema_matcher_can_be_frozen_for_production_p2() -> None:
    source = make_model(seed=40)
    target = QSREProductionOperatorInducerV3(cfg())
    target.load_pretrained_schema_matcher(
        source.schema_matcher.state_dict(),
        freeze=True,
    )
    assert all(
        not parameter.requires_grad
        for parameter in target.schema_matcher.parameters()
    )


def test_relation_hypotheses_remain_continuous_until_binding() -> None:
    model = make_model(seed=50)
    q, qmask = query(seed=51)
    schema = relation_schema(6, seed=52)
    relation = run(
        model,
        q=q,
        qmask=qmask,
        schema=schema,
        max_steps=3,
    )["operator"].relation_distribution
    assert torch.isfinite(relation).all()
    assert torch.all(relation > 0)
    assert torch.allclose(
        relation.sum(dim=-1),
        torch.ones_like(relation.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_ordered_query_evidence_coverage_is_differentiable_and_monotone() -> None:
    model = make_model(seed=60)
    with torch.no_grad():
        model.continue_head.weight.zero_()
        model.stop_head.weight.zero_()
        model.unknown_head.weight.zero_()
        model.continue_head.bias.fill_(8.0)
        model.stop_head.bias.fill_(-8.0)
        model.unknown_head.bias.fill_(-8.0)
        model.event_match_projection.weight.zero_()
        model.event_match_projection.bias.zero_()

    q, qmask = query(batch=1, seed=61)
    schema = relation_schema(4, seed=62)
    coverage = run(
        model,
        q=q,
        qmask=qmask,
        schema=schema,
        max_steps=3,
    )["query_coverage"]
    assert coverage.shape == (1, 3, q.size(2))
    assert torch.all(coverage[:, 1:] + 1e-7 >= coverage[:, :-1])
    assert float(coverage[:, -1].max()) > 0.0


def test_schema_permutation_equivariance_survives_ordered_coverage() -> None:
    model = make_model(seed=70)
    q, qmask = query(batch=1, seed=71)
    base_schema = relation_schema(5, seed=72)
    base = run(
        model,
        q=q,
        qmask=qmask,
        schema=base_schema,
        max_steps=3,
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
        schema=moved_schema,
        max_steps=3,
    )["operator"].relation_distribution.detach()
    assert torch.allclose(
        base,
        moved[..., inverse],
        atol=1e-5,
        rtol=1e-5,
    )


def test_p1_schema_relation_state_is_not_match_authority() -> None:
    model = make_model(seed=80)
    q, qmask = query(batch=1, seed=81)
    schema = relation_schema(4, seed=82)
    summary = schema.token_states.mean(dim=(1, 2))
    a = run(
        model,
        q=q,
        qmask=qmask,
        schema=schema,
        schema_relation_state=summary,
    )["operator"]
    b = run(
        model,
        q=q,
        qmask=qmask,
        schema=schema,
        schema_relation_state=torch.randn_like(summary) * 1000.0,
    )["operator"]
    assert torch.allclose(
        a.relation_distribution,
        b.relation_distribution,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a.continuous_state,
        b.continuous_state,
        atol=1e-6,
        rtol=1e-6,
    )


def test_factor_semantics_do_not_depend_on_relation_cardinality() -> None:
    model = make_model(seed=90)
    q, qmask = query(batch=1, seed=91)
    small = relation_schema(3, seed=92)
    extra = relation_schema(2, seed=93)
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
    a = run(model, q=q, qmask=qmask, schema=small, max_steps=1)["operator"]
    b = run(model, q=q, qmask=qmask, schema=large, max_steps=1)["operator"]
    for name in (
        "role_distribution",
        "traversal_distribution",
        "direction_distribution",
        "modifier_weight",
        "control_distribution",
    ):
        assert torch.allclose(
            getattr(a, name),
            getattr(b, name),
            atol=1e-6,
            rtol=1e-6,
        )


def test_unknown_and_stop_remain_distinct_events() -> None:
    model = make_model(seed=100)
    q, qmask = query(batch=1, seed=101)
    schema = relation_schema(3, seed=102)
    with torch.no_grad():
        model.continue_head.weight.zero_()
        model.stop_head.weight.zero_()
        model.unknown_head.weight.zero_()
        model.continue_head.bias.fill_(-30.0)
        model.stop_head.bias.fill_(30.0)
        model.unknown_head.bias.fill_(-30.0)
        model.event_match_projection.weight.zero_()
        model.event_match_projection.bias.zero_()
    stopped = run(
        model,
        q=q,
        qmask=qmask,
        schema=schema,
        max_steps=1,
    )["event_distribution"]

    with torch.no_grad():
        model.stop_head.bias.fill_(-30.0)
        model.unknown_head.bias.fill_(30.0)
    unknown = run(
        model,
        q=q,
        qmask=qmask,
        schema=schema,
        max_steps=1,
    )["event_distribution"]

    assert float(stopped[0, 0, EVENT_STOP]) > 0.999
    assert float(stopped[0, 0, EVENT_UNKNOWN]) < 1.0e-6
    assert float(unknown[0, 0, EVENT_UNKNOWN]) > 0.999
    assert float(unknown[0, 0, EVENT_STOP]) < 1.0e-6
