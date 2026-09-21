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
        "schema": "alice.eipm.n0.qsre-frozen-factor-schema-cache.v3",
        "private_identity_data": False,
        "gradient": False,
        "optimizer": False,
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


def authority(
    *,
    batch: int,
    relations: int,
    seed: int = 200,
) -> tuple[dict, dict]:
    g = torch.Generator().manual_seed(seed)

    def pair(count: int) -> dict:
        return {
            "joint_preference": torch.randn(batch, count, generator=g),
            "semantic_projection": torch.randn(batch, count, generator=g),
        }

    relation = pair(relations)
    factors = {
        "role": pair(4),
        "traversal": pair(3),
        "direction": pair(3),
        "control": pair(3),
        "modifiers": {
            "joint_preference": torch.randn(batch, 4, 2, generator=g),
            "semantic_projection": torch.randn(batch, 4, 2, generator=g),
        },
    }
    return relation, factors


def run(
    model: QSREProductionOperatorInducerV3,
    *,
    q: torch.Tensor,
    qmask: torch.Tensor,
    schema: QSREDynamicRelationSchema,
    max_steps: int = 2,
    schema_relation_state: torch.Tensor | None = None,
    relation_authority: dict | None = None,
    factor_authority: dict | None = None,
):
    if schema_relation_state is None:
        schema_relation_state = schema.token_states.mean(dim=(1, 2))
    if relation_authority is None or factor_authority is None:
        relation_authority, factor_authority = authority(
            batch=q.size(0),
            relations=schema.token_states.size(0),
        )
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
        relation_authority=relation_authority,
        factor_authority=factor_authority,
        max_steps=max_steps,
    )


def test_operator_uses_external_frozen_authority_and_no_trainable_matcher() -> None:
    model = make_model()
    report = model.parameter_report()
    assert report["fixed_factor_class_head_parameters"] == 0
    assert report["schema_matcher_trainable_parameters"] == 0
    assert report["frozen_semantic_authority_external"] is True
    assert report["frozen_authority_trainable_parameters"] == 0
    assert report["relation_identity_owned_by_trainable_operator"] is False
    assert report["factor_identity_owned_by_trainable_operator"] is False
    assert report["ordered_query_evidence_coverage"] is True
    assert report["exact_sparsity_owned_by_binder"] is True


def test_changing_only_frozen_relation_authority_changes_relation_output() -> None:
    model = make_model(seed=40)
    q, qmask = query(batch=1, seed=41)
    schema = relation_schema(3, seed=42)
    rel, fac = authority(batch=1, relations=3, seed=43)
    with torch.no_grad():
        model.step_query.weight.zero_()
    first = {k: v.clone() for k, v in rel.items()}
    second = {k: v.clone() for k, v in rel.items()}
    first["joint_preference"][:] = torch.tensor([[8.0, -2.0, -2.0]])
    first["semantic_projection"][:] = torch.tensor([[8.0, -2.0, -2.0]])
    second["joint_preference"][:] = torch.tensor([[-2.0, 8.0, -2.0]])
    second["semantic_projection"][:] = torch.tensor([[-2.0, 8.0, -2.0]])
    a = run(
        model, q=q, qmask=qmask, schema=schema, max_steps=1,
        relation_authority=first, factor_authority=fac,
    )["operator"].relation_distribution
    b = run(
        model, q=q, qmask=qmask, schema=schema, max_steps=1,
        relation_authority=second, factor_authority=fac,
    )["operator"].relation_distribution
    assert int(a[0, 0].argmax()) == 0
    assert int(b[0, 0].argmax()) == 1


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


def test_schema_permutation_equivariance_requires_same_authority_permutation() -> None:
    model = make_model(seed=70)
    q, qmask = query(batch=1, seed=71)
    base_schema = relation_schema(5, seed=72)
    rel, fac = authority(batch=1, relations=5, seed=73)
    base = run(
        model, q=q, qmask=qmask, schema=base_schema, max_steps=2,
        relation_authority=rel, factor_authority=fac,
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
    moved_rel = {key: value[:, perm] for key, value in rel.items()}
    moved = run(
        model, q=q, qmask=qmask, schema=moved_schema, max_steps=2,
        relation_authority=moved_rel, factor_authority=fac,
    )["operator"].relation_distribution.detach()
    assert torch.allclose(
        base,
        moved[..., inverse],
        atol=1e-5,
        rtol=1e-5,
    )


def test_p1_schema_relation_state_remains_interface_only() -> None:
    model = make_model(seed=80)
    q, qmask = query(batch=1, seed=81)
    schema = relation_schema(4, seed=82)
    rel, fac = authority(batch=1, relations=4, seed=83)
    summary = schema.token_states.mean(dim=(1, 2))
    a = run(
        model, q=q, qmask=qmask, schema=schema,
        schema_relation_state=summary,
        relation_authority=rel, factor_authority=fac,
    )["operator"]
    b = run(
        model, q=q, qmask=qmask, schema=schema,
        schema_relation_state=torch.randn_like(summary) * 1000.0,
        relation_authority=rel, factor_authority=fac,
    )["operator"]
    assert torch.allclose(a.relation_distribution, b.relation_distribution)
    assert torch.allclose(a.continuous_state, b.continuous_state)


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
    stopped = run(model, q=q, qmask=qmask, schema=schema, max_steps=1)[
        "event_distribution"
    ]

    with torch.no_grad():
        model.stop_head.bias.fill_(-30.0)
        model.unknown_head.bias.fill_(30.0)
    unknown = run(model, q=q, qmask=qmask, schema=schema, max_steps=1)[
        "event_distribution"
    ]

    assert float(stopped[0, 0, EVENT_STOP]) > 0.999
    assert float(stopped[0, 0, EVENT_UNKNOWN]) < 1.0e-6
    assert float(unknown[0, 0, EVENT_UNKNOWN]) > 0.999
    assert float(unknown[0, 0, EVENT_STOP]) < 1.0e-6


def test_factor_cardinality_is_runtime_config_not_parameter_topology() -> None:
    base = QSREProductionOperatorInducerV3(cfg())
    expanded_cfg = QSREProductionConfig(
        semantic_dim=24,
        model_dim=24,
        num_hidden_states=3,
        num_attention_heads=4,
        operator_refinement_layers=1,
        field_state_dim=24,
        field_metadata_dim=3,
        role_count=6,
        traversal_count=5,
        direction_count=4,
        modifier_count=7,
        control_count=5,
        dropout=0.0,
    )
    expanded = QSREProductionOperatorInducerV3(expanded_cfg)
    assert sum(p.numel() for p in base.parameters()) == sum(
        p.numel() for p in expanded.parameters()
    )

    g = torch.Generator().manual_seed(333)

    def categorical(count: int) -> dict:
        return {
            "keys": [f"x{i}" for i in range(count)],
            "token_states": torch.randn(count, 3, 4, 24, generator=g),
            "token_mask": torch.ones(count, 4, dtype=torch.bool),
        }

    cache = {
        "schema": "alice.eipm.n0.qsre-frozen-factor-schema-cache.v3",
        "private_identity_data": False,
        "gradient": False,
        "optimizer": False,
        "role": categorical(6),
        "traversal": categorical(5),
        "direction": categorical(4),
        "control": categorical(5),
        "modifiers": {
            "keys": [f"m{i}" for i in range(7)],
            "token_states": torch.randn(7, 2, 3, 4, 24, generator=g),
            "token_mask": torch.ones(7, 2, 4, dtype=torch.bool),
        },
    }
    expanded.configure_factor_schema_cache(cache)
    report = expanded.parameter_report()
    assert report["role_count_ceiling"] is None
    assert report["traversal_count_ceiling"] is None
    assert report["direction_count_ceiling"] is None
    assert report["control_count_ceiling"] is None
