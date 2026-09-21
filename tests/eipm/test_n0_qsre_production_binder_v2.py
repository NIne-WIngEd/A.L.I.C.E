from __future__ import annotations

import torch

from alice_personality.n0.qsre_production_binder_v2 import (
    QSREProductionBinderV2,
)
from alice_personality.n0.qsre_production_core import (
    CONTROL_RELATIONAL,
    DIRECTION_FORWARD,
    ROLE_TARGET,
    TRAVERSAL_PATH,
    QSREDynamicRelationSchema,
    QSREProductionConfig,
    QSREProductionOperatorState,
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


def schema(relations: int = 3) -> QSREDynamicRelationSchema:
    g = torch.Generator().manual_seed(1)
    return QSREDynamicRelationSchema(
        token_states=torch.randn(relations, 3, 5, 24, generator=g),
        token_mask=torch.ones(relations, 5, dtype=torch.bool),
        domain_type_mask=torch.ones(relations, 4, dtype=torch.bool),
        range_type_mask=torch.ones(relations, 4, dtype=torch.bool),
        symmetric=torch.zeros(relations, dtype=torch.bool),
    )


def operator(
    *,
    relation_count: int = 3,
    selected_relation: int = 0,
) -> QSREProductionOperatorState:
    relation = torch.zeros(1, 1, relation_count)
    relation[0, 0, selected_relation] = 1.0
    role = torch.zeros(1, 4)
    role[0, ROLE_TARGET] = 1.0
    traversal = torch.zeros(1, 3)
    traversal[0, TRAVERSAL_PATH] = 1.0
    direction = torch.zeros(1, 3)
    direction[0, DIRECTION_FORWARD] = 1.0
    control = torch.zeros(1, 3)
    control[0, CONTROL_RELATIONAL] = 1.0
    return QSREProductionOperatorState(
        relation_distribution=relation,
        relation_step_mass=torch.ones(1, 1),
        stop_probability=torch.zeros(1, 1),
        unknown_probability=torch.zeros(1, 1),
        role_distribution=role,
        traversal_distribution=traversal,
        direction_distribution=direction,
        modifier_weight=torch.zeros(1, 4),
        applicability=torch.ones(1),
        control_distribution=control,
        continuous_state=torch.zeros(1, 24),
        uncertainty=torch.zeros(1),
    )


def inputs():
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, 3, 7, 24, generator=g)
    qmask = torch.ones(1, 7, dtype=torch.bool)
    fields = torch.randn(1, 4, 6, 24, generator=g)
    fmask = torch.ones(1, 4, 6, dtype=torch.bool)
    edge_index = torch.tensor([[[0, 1], [0, 2], [2, 3]]])
    edge_relation = torch.tensor([[0, 1, 2]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    return dict(
        query_hidden_states=q,
        query_token_mask=qmask,
        field_token_states=fields,
        field_token_mask=fmask,
        field_type_id=torch.zeros(1, 4, dtype=torch.long),
        edge_index=edge_index,
        edge_relation_index=edge_relation,
        edge_valid_mask=valid,
        edge_reliability=torch.ones(1, 3),
        edge_recency=torch.ones(1, 3),
    )


def test_binder_v2_p1_schema_state_is_not_match_authority() -> None:
    torch.manual_seed(3)
    binder = QSREProductionBinderV2(cfg()).eval()
    s = schema()
    base = inputs()
    op = operator()
    a = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=op,
    )
    b = binder(
        **base,
        schema=s,
        schema_relation_state=torch.randn(3, 24) * 1000.0,
        operator=op,
    )
    for key in (
        "support_logits",
        "edge_support_weight",
        "focus_logits",
        "focus_field_weight",
    ):
        assert torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-6)


def test_binder_v2_operator_relation_mass_controls_support_semantics() -> None:
    torch.manual_seed(4)
    binder = QSREProductionBinderV2(cfg()).eval()
    s = schema()
    base = inputs()
    a = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=operator(selected_relation=0),
    )
    b = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=operator(selected_relation=1),
    )
    assert not torch.allclose(
        a["support_logits"],
        b["support_logits"],
    )


def test_binder_v2_operator_continuous_state_is_not_support_authority() -> None:
    torch.manual_seed(40)
    binder = QSREProductionBinderV2(cfg()).eval()
    s = schema()
    base = inputs()
    op_a = operator()
    op_b = QSREProductionOperatorState(
        **{
            **op_a.__dict__,
            "continuous_state": torch.randn_like(op_a.continuous_state) * 1000.0,
        }
    )
    a = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=op_a,
    )
    b = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=op_b,
    )
    assert torch.allclose(
        a["support_logits"],
        b["support_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a["edge_support_weight"],
        b["edge_support_weight"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_binder_v2_type_contradiction_is_exact_zero() -> None:
    torch.manual_seed(5)
    binder = QSREProductionBinderV2(cfg()).eval()
    s = schema(2)
    domain = torch.zeros_like(s.domain_type_mask)
    range_mask = torch.zeros_like(s.range_type_mask)
    domain[0, 0] = True
    range_mask[0, 1] = True
    s = QSREDynamicRelationSchema(
        token_states=s.token_states,
        token_mask=s.token_mask,
        domain_type_mask=domain,
        range_type_mask=range_mask,
        symmetric=s.symmetric,
    )
    base = inputs()
    base["edge_index"] = torch.tensor([[[0, 1], [0, 2]]])
    base["edge_relation_index"] = torch.tensor([[0, 0]])
    base["edge_valid_mask"] = torch.ones(1, 2, dtype=torch.bool)
    base["edge_reliability"] = torch.ones(1, 2)
    base["edge_recency"] = torch.ones(1, 2)
    base["field_type_id"] = torch.zeros(1, 4, dtype=torch.long)
    out = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(2, 24),
        operator=operator(relation_count=2, selected_relation=0),
    )
    assert torch.equal(
        out["edge_support_weight"],
        torch.zeros_like(out["edge_support_weight"]),
    )


def test_binder_v2_focus_is_query_field_only_and_total() -> None:
    torch.manual_seed(6)
    binder = QSREProductionBinderV2(cfg()).eval()
    s = schema()
    base = inputs()
    base["field_token_states"].zero_()
    base["field_token_states"][0, 2, :6] = base[
        "query_hidden_states"
    ][0, -1, :6]
    out = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=operator(),
    )
    assert torch.isfinite(out["focus_field_weight"]).all()
    assert int(out["focus_field_weight"].argmax(dim=-1).item()) == 2


def test_binder_v2_edge_permutation_equivariance() -> None:
    torch.manual_seed(7)
    binder = QSREProductionBinderV2(cfg()).eval()
    s = schema()
    base = inputs()
    op = operator()
    a = binder(
        **base,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=op,
    )["edge_support_weight"]
    perm = torch.tensor([2, 0, 1])
    moved = dict(base)
    for key in (
        "edge_index",
        "edge_relation_index",
        "edge_valid_mask",
        "edge_reliability",
        "edge_recency",
    ):
        moved[key] = moved[key][:, perm]
    b = binder(
        **moved,
        schema=s,
        schema_relation_state=torch.zeros(3, 24),
        operator=op,
    )["edge_support_weight"]
    assert torch.allclose(
        a,
        b[:, torch.argsort(perm)],
        atol=1e-6,
        rtol=1e-6,
    )


def test_binder_v2_parameter_report_has_no_support_ceiling() -> None:
    report = QSREProductionBinderV2(cfg()).parameter_report()
    assert report["relation_count_dependent_parameters"] == 0
    assert report["relation_semantics_owned_by_operator_relation_mass"] is True
    assert report["operator_continuous_state_not_support_authority"] is True
    assert report["p1_schema_relation_state_not_match_authority"] is True
    assert report["shared_query_field_metric"] is True
    assert report["fixed_top_k"] is False
    assert report["field_count_ceiling"] is None
    assert report["edge_count_ceiling"] is None
    assert report["support_count_ceiling"] is None
    assert report["focus_field_count_ceiling"] is None
