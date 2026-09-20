from __future__ import annotations

import copy

import torch

from alice_personality.n0.qsre_production_core import (
    CONTROL_RELATIONAL,
    DIRECTION_BIDIRECTIONAL,
    DIRECTION_FORWARD,
    DIRECTION_REVERSE,
    MOD_RECENCY,
    MOD_RELIABILITY,
    ROLE_SOURCE,
    ROLE_TARGET,
    TRAVERSAL_AGGREGATE,
    TRAVERSAL_LOCAL,
    TRAVERSAL_PATH,
    QSREDynamicRelationSchema,
    QSREProductionBinder,
    QSREProductionConfig,
    QSREProductionCore,
    QSREProductionExecutor,
    QSREProductionOperatorInducer,
    QSREProductionSchemaEncoder,
    QSREProductionOperatorState,
)


def cfg() -> QSREProductionConfig:
    return QSREProductionConfig(
        semantic_dim=24,
        model_dim=32,
        num_hidden_states=3,
        num_attention_heads=4,
        operator_refinement_layers=1,
        field_state_dim=24,
        field_metadata_dim=3,
        direction_count=3,
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




def run_operator(
    model: QSREProductionOperatorInducer,
    encoder: QSREProductionSchemaEncoder,
    *,
    q: torch.Tensor,
    qmask: torch.Tensor,
    s: QSREDynamicRelationSchema,
    steps: int,
):
    encoded = encoder(s)
    return model(
        query_hidden_states=q,
        query_token_mask=qmask,
        schema=s,
        schema_token_state=encoded["schema_token_state"],
        schema_relation_state=encoded["schema_relation_state"],
        max_steps=steps,
    )

def query(batch: int = 2, tokens: int = 7, *, seed: int = 2):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, 3, tokens, 24, generator=g)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return hidden, mask


def one_hot_operator(
    *,
    batch: int,
    relation_count: int,
    relation_sequence: list[int],
    traversal: int = TRAVERSAL_LOCAL,
    direction: int = DIRECTION_FORWARD,
    applicability: float = 1.0,
    role: int = ROLE_TARGET,
    modifiers: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    continuous: torch.Tensor | None = None,
) -> QSREProductionOperatorState:
    steps = len(relation_sequence)
    relation = torch.zeros(batch, steps, relation_count)
    for step, rid in enumerate(relation_sequence):
        relation[:, step, rid] = 1.0
    step_mass = torch.ones(batch, steps)
    role_distribution = torch.zeros(batch, 4)
    role_distribution[:, role] = 1.0
    traversal_distribution = torch.zeros(batch, 3)
    traversal_distribution[:, traversal] = 1.0
    direction_distribution = torch.zeros(batch, 3)
    direction_distribution[:, direction] = 1.0
    modifier_weight = torch.tensor(modifiers).view(1, 4).expand(batch, -1).clone()
    control = torch.zeros(batch, 3)
    control[:, CONTROL_RELATIONAL] = 1.0
    if continuous is None:
        continuous = torch.zeros(batch, 32)
    return QSREProductionOperatorState(
        relation_distribution=relation,
        relation_step_mass=step_mass,
        stop_probability=torch.zeros(batch, steps),
        unknown_probability=torch.zeros(batch, steps),
        role_distribution=role_distribution,
        traversal_distribution=traversal_distribution,
        direction_distribution=direction_distribution,
        modifier_weight=modifier_weight,
        applicability=torch.full((batch,), float(applicability)),
        control_distribution=control,
        continuous_state=continuous,
        uncertainty=torch.zeros(batch),
    )


def graph_inputs(batch: int = 1):
    field_state = torch.randn(batch, 4, 24, generator=torch.Generator().manual_seed(30))
    field_metadata = torch.zeros(batch, 4, 3)
    field_valid = torch.ones(batch, 4, dtype=torch.bool)
    edge_index = torch.tensor([[[0, 1], [1, 2], [0, 3]]], dtype=torch.long).expand(batch, -1, -1).clone()
    edge_relation = torch.tensor([[0, 1, 1]], dtype=torch.long).expand(batch, -1).clone()
    edge_valid = torch.ones(batch, 3, dtype=torch.bool)
    support = torch.ones(batch, 3)
    reliability = torch.tensor([[0.8, 0.7, 0.2]]).expand(batch, -1).clone()
    recency = torch.tensor([[0.3, 0.9, 0.4]]).expand(batch, -1).clone()
    return {
        "field_state": field_state,
        "field_metadata": field_metadata,
        "field_valid_mask": field_valid,
        "edge_index": edge_index,
        "edge_relation_index": edge_relation,
        "edge_valid_mask": edge_valid,
        "edge_support_weight": support,
        "edge_reliability": reliability,
        "edge_recency": recency,
        "edge_temporal_match": torch.ones_like(support),
        "edge_provenance_match": torch.ones_like(support),
    }


def test_dynamic_relation_cardinality_has_no_parameter_axis() -> None:
    torch.manual_seed(4)
    model = QSREProductionOperatorInducer(cfg()).eval()
    encoder = QSREProductionSchemaEncoder(cfg()).eval()
    before = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in encoder.parameters())
    q, qmask = query(batch=2)

    for relations in (3, 6, 11):
        out = run_operator(
            model,
            encoder,
            q=q,
            qmask=qmask,
            s=schema(relations, seed=relations),
            steps=2,
        )
        operator = out["operator"]
        assert operator.relation_distribution.shape == (2, 2, relations)
        assert out["schema_relation_state"].shape == (relations, 32)
        assert sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in encoder.parameters()) == before

    report = model.parameter_report()
    assert report["relation_count_dependent_parameters"] == 0
    assert report["runtime_dynamic_relation_schema"] is True


def test_dynamic_step_count_uses_shared_parameters() -> None:
    torch.manual_seed(5)
    model = QSREProductionOperatorInducer(cfg()).eval()
    encoder = QSREProductionSchemaEncoder(cfg()).eval()
    q, qmask = query(batch=1)
    s = schema(6)
    before = {name: id(parameter) for name, parameter in model.named_parameters()}

    for steps in (1, 2, 3, 5):
        out = run_operator(
            model,
            encoder,
            q=q,
            qmask=qmask,
            s=s,
            steps=steps,
        )
        assert out["operator"].relation_distribution.shape == (1, steps, 6)
        assert {name: id(parameter) for name, parameter in model.named_parameters()} == before

    assert model.parameter_report()["hop_count_dependent_parameters"] == 0


def test_schema_permutation_equivariance() -> None:
    torch.manual_seed(6)
    model = QSREProductionOperatorInducer(cfg()).eval()
    encoder = QSREProductionSchemaEncoder(cfg()).eval()
    q, qmask = query(batch=2, seed=7)
    base_schema = schema(6, seed=8)
    base = run_operator(
        model,
        encoder,
        q=q,
        qmask=qmask,
        s=base_schema,
        steps=3,
    )["operator"].relation_distribution.detach()

    perm = torch.tensor([2, 5, 0, 4, 1, 3])
    inverse = torch.argsort(perm)
    permuted_schema = QSREDynamicRelationSchema(
        token_states=base_schema.token_states[perm],
        token_mask=base_schema.token_mask[perm],
        domain_type_mask=base_schema.domain_type_mask[perm],
        range_type_mask=base_schema.range_type_mask[perm],
        symmetric=base_schema.symmetric[perm],
    )
    moved = run_operator(
        model,
        encoder,
        q=q,
        qmask=qmask,
        s=permuted_schema,
        steps=3,
    )["operator"].relation_distribution.detach()

    assert torch.allclose(base, moved[..., inverse], atol=1e-5, rtol=1e-5)


def test_semantic_schema_intervention_changes_relation_state_and_logits() -> None:
    torch.manual_seed(9)
    model = QSREProductionOperatorInducer(cfg()).eval()
    encoder = QSREProductionSchemaEncoder(cfg()).eval()
    q, qmask = query(batch=1, seed=10)
    s1 = schema(4, seed=11)
    s2 = QSREDynamicRelationSchema(
        token_states=s1.token_states.clone(),
        token_mask=s1.token_mask.clone(),
        domain_type_mask=s1.domain_type_mask.clone(),
        range_type_mask=s1.range_type_mask.clone(),
        symmetric=s1.symmetric.clone(),
    )
    s2.token_states[0].mul_(-3.0)

    a = run_operator(model, encoder, q=q, qmask=qmask, s=s1, steps=1)
    b = run_operator(model, encoder, q=q, qmask=qmask, s=s2, steps=1)

    assert not torch.allclose(
        a["schema_relation_state"][0],
        b["schema_relation_state"][0],
    )
    assert not torch.allclose(
        a["operator"].relation_distribution[..., 0],
        b["operator"].relation_distribution[..., 0],
    )


def test_unknown_and_stop_are_distinct_structural_outputs() -> None:
    torch.manual_seed(12)
    model = QSREProductionOperatorInducer(cfg()).eval()
    encoder = QSREProductionSchemaEncoder(cfg()).eval()
    q, qmask = query(batch=1)
    s = schema(3)

    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.unknown_head.weight.zero_()
        model.stop_head.bias.fill_(20.0)
        model.unknown_head.bias.fill_(-20.0)
    stopped = run_operator(
        model, encoder, q=q, qmask=qmask, s=s, steps=1
    )["operator"]

    with torch.no_grad():
        model.stop_head.bias.fill_(-20.0)
        model.unknown_head.bias.fill_(20.0)
    unknown = run_operator(
        model, encoder, q=q, qmask=qmask, s=s, steps=1
    )["operator"]

    assert float(stopped.stop_probability[0, 0]) > 0.99
    assert float(stopped.unknown_probability[0, 0]) < 0.01
    assert float(unknown.unknown_probability[0, 0]) > 0.99
    assert float(unknown.stop_probability[0, 0]) < 0.01


def test_type_domain_range_contradiction_zeroes_support() -> None:
    torch.manual_seed(13)
    binder = QSREProductionBinder(cfg()).eval()
    s = schema(2)
    # Relation 0 requires source type 0 and target type 1 only.
    domain = torch.zeros_like(s.domain_type_mask)
    range_mask = torch.zeros_like(s.range_type_mask)
    domain[0, 0] = True
    range_mask[0, 1] = True
    # Relation 1 accepts source type 2 -> target type 3.
    domain[1, 2] = True
    range_mask[1, 3] = True
    s = QSREDynamicRelationSchema(
        token_states=s.token_states,
        token_mask=s.token_mask,
        domain_type_mask=domain,
        range_type_mask=range_mask,
        symmetric=s.symmetric,
    )

    q, qmask = query(batch=1)
    fields = torch.randn(1, 3, 4, 24)
    field_mask = torch.ones(1, 3, 4, dtype=torch.bool)
    field_types = torch.tensor([[2, 3, 1]])
    edge_index = torch.tensor([[[0, 1], [0, 2]]])
    edge_rel = torch.tensor([[0, 1]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    op = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0],
    )
    out = binder(
        query_hidden_states=q,
        query_token_mask=qmask,
        field_token_states=fields,
        field_token_mask=field_mask,
        field_type_id=field_types,
        edge_index=edge_index,
        edge_relation_index=edge_rel,
        edge_valid_mask=valid,
        edge_reliability=torch.ones(1, 2),
        edge_recency=torch.ones(1, 2),
        schema=s,
        schema_relation_state=torch.randn(2, 32),
        operator=op,
    )

    # Relation-0 edge is lexically/operator favored but structurally impossible.
    assert out["type_compatible"].tolist() == [[False, False]]
    assert torch.equal(out["edge_support_weight"], torch.zeros_like(out["edge_support_weight"]))


def test_plural_relation_hypotheses_are_representable_without_argmax() -> None:
    op = one_hot_operator(
        batch=1,
        relation_count=3,
        relation_sequence=[0],
    )
    relation = op.relation_distribution.clone()
    relation[0, 0] = torch.tensor([0.5, 0.5, 0.0])
    plural = QSREProductionOperatorState(
        **{**op.__dict__, "relation_distribution": relation}
    )
    plural.validate(relation_count=3, model_dim=32)
    assert torch.count_nonzero(plural.relation_distribution[0, 0]).item() == 2


def test_composable_traversal_and_arbitration_factors() -> None:
    op = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0, 1],
        traversal=TRAVERSAL_PATH,
        modifiers=(1.0, 1.0, 0.0, 0.0),
    )
    assert float(op.traversal_distribution[0, TRAVERSAL_PATH]) == 1.0
    assert float(op.modifier_weight[0, MOD_RELIABILITY]) == 1.0
    assert float(op.modifier_weight[0, MOD_RECENCY]) == 1.0


def test_continuous_operator_state_changes_executor_state() -> None:
    torch.manual_seed(14)
    executor = QSREProductionExecutor(cfg()).eval()
    g = graph_inputs()
    schema_state = torch.randn(2, 32)
    base = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0],
        continuous=torch.zeros(1, 32),
    )
    changed = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0],
        continuous=torch.ones(1, 32),
    )

    out_a = executor(
        **g,
        schema_relation_state=schema_state,
        operator=base,
        focus_field_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    out_b = executor(
        **g,
        schema_relation_state=schema_state,
        operator=changed,
        focus_field_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert not torch.allclose(out_a["node_state"], out_b["node_state"])


def test_zero_one_many_support_are_total() -> None:
    torch.manual_seed(15)
    executor = QSREProductionExecutor(cfg()).eval()
    base = graph_inputs()
    schema_state = torch.randn(2, 32)
    op = one_hot_operator(batch=1, relation_count=2, relation_sequence=[0])

    for support in (
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0]]),
    ):
        g = dict(base)
        g["edge_support_weight"] = support
        out = executor(
            **g,
            schema_relation_state=schema_state,
            operator=op,
            focus_field_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
        assert torch.isfinite(out["relational_probability"]).all()
        if float(support.sum()) == 0.0:
            assert float(out["relational_probability"].sum()) == 0.0


def test_multi_hop_relation_order_changes_path_execution() -> None:
    torch.manual_seed(16)
    executor = QSREProductionExecutor(cfg()).eval()
    g = graph_inputs()
    schema_state = torch.randn(2, 32)
    focus = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    forward = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0, 1],
        traversal=TRAVERSAL_PATH,
    )
    reverse = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[1, 0],
        traversal=TRAVERSAL_PATH,
    )
    a = executor(
        **g,
        schema_relation_state=schema_state,
        operator=forward,
        focus_field_weight=focus,
    )
    b = executor(
        **g,
        schema_relation_state=schema_state,
        operator=reverse,
        focus_field_weight=focus,
    )
    assert not torch.allclose(a["path_frontier"], b["path_frontier"])


def test_edge_and_field_permutation_equivariance() -> None:
    torch.manual_seed(17)
    executor = QSREProductionExecutor(cfg()).eval()
    g = graph_inputs()
    schema_state = torch.randn(2, 32)
    op = one_hot_operator(batch=1, relation_count=2, relation_sequence=[0])
    focus = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    base = executor(
        **g,
        schema_relation_state=schema_state,
        operator=op,
        focus_field_weight=focus,
    )["relational_probability"]

    field_perm = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(field_perm)
    old_to_new = torch.empty_like(field_perm)
    old_to_new[field_perm] = torch.arange(4)

    gp = dict(g)
    gp["field_state"] = g["field_state"][:, field_perm]
    gp["field_metadata"] = g["field_metadata"][:, field_perm]
    gp["field_valid_mask"] = g["field_valid_mask"][:, field_perm]
    gp["edge_index"] = old_to_new[g["edge_index"]]

    edge_perm = torch.tensor([2, 0, 1])
    for key in (
        "edge_index",
        "edge_relation_index",
        "edge_valid_mask",
        "edge_support_weight",
        "edge_reliability",
        "edge_recency",
        "edge_temporal_match",
        "edge_provenance_match",
    ):
        gp[key] = gp[key][:, edge_perm]

    focus_perm = focus[:, field_perm]
    moved = executor(
        **gp,
        schema_relation_state=schema_state,
        operator=op,
        focus_field_weight=focus_perm,
    )["relational_probability"]

    assert torch.allclose(base, moved[:, inverse], atol=1e-5, rtol=1e-5)


def test_nonrelational_applicability_is_exact_zero_output() -> None:
    torch.manual_seed(18)
    executor = QSREProductionExecutor(cfg()).eval()
    g = graph_inputs()
    schema_state = torch.randn(2, 32)
    op = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0],
        applicability=0.0,
    )
    out = executor(
        **g,
        schema_relation_state=schema_state,
        operator=op,
        focus_field_weight=torch.zeros(1, 4),
    )
    assert torch.equal(
        out["relational_probability"],
        torch.zeros_like(out["relational_probability"]),
    )


def test_totality_fuzz_over_operator_space() -> None:
    torch.manual_seed(19)
    executor = QSREProductionExecutor(cfg()).eval()
    g = graph_inputs(batch=4)
    schema_state = torch.randn(3, 32)

    for steps in (1, 2, 4):
        relation = torch.rand(4, steps, 3)
        relation = relation / relation.sum(dim=-1, keepdim=True)
        mass = torch.rand(4, steps)
        role = torch.rand(4, 4)
        role = role / role.sum(dim=-1, keepdim=True)
        traversal = torch.rand(4, 3)
        traversal = traversal / traversal.sum(dim=-1, keepdim=True)
        direction = torch.rand(4, 3)
        direction = direction / direction.sum(dim=-1, keepdim=True)
        modifiers = torch.rand(4, 4)
        control = torch.rand(4, 3)
        control = control / control.sum(dim=-1, keepdim=True)
        op = QSREProductionOperatorState(
            relation_distribution=relation,
            relation_step_mass=mass,
            stop_probability=torch.rand(4, steps),
            unknown_probability=torch.rand(4, steps),
            role_distribution=role,
            traversal_distribution=traversal,
            direction_distribution=direction,
            modifier_weight=modifiers,
            applicability=torch.rand(4),
            control_distribution=control,
            continuous_state=torch.randn(4, 32),
            uncertainty=torch.rand(4),
        )
        out = executor(
            **g,
            schema_relation_state=schema_state,
            operator=op,
            focus_field_weight=torch.zeros(4, 4),
        )
        assert torch.isfinite(out["relational_probability"]).all()
        assert torch.isfinite(out["relational_summary"]).all()


def test_production_core_parameter_report_has_no_relation_or_hop_parameter_axis() -> None:
    core = QSREProductionCore(cfg())
    report = core.parameter_report()
    assert report["schema_encoder"]["relation_count_dependent_parameters"] == 0
    assert report["operator"]["relation_count_dependent_parameters"] == 0
    assert report["operator"]["hop_count_dependent_parameters"] == 0
    assert report["executor"]["relation_count_dependent_parameters"] == 0
    assert report["executor"]["hop_count_dependent_parameters"] == 0
    assert report["private_identity_parameters"] == 0


def test_checkpoint_factor_counts_are_not_hard_runtime_ceilings() -> None:
    expanded = QSREProductionConfig(
        semantic_dim=24,
        model_dim=32,
        num_hidden_states=3,
        num_attention_heads=4,
        operator_refinement_layers=1,
        field_state_dim=24,
        field_metadata_dim=3,
        role_count=6,
        traversal_count=5,
        direction_count=5,
        modifier_count=7,
        control_count=4,
        dropout=0.0,
    )
    expanded.validate()
    operator = QSREProductionOperatorInducer(expanded)
    report = operator.parameter_report()
    assert report["role_count_ceiling"] is None
    assert report["traversal_count_ceiling"] is None
    assert report["direction_count_ceiling"] is None
    assert report["control_count_ceiling"] is None


def test_schema_encoder_starts_semantic_preserving_not_random_role_ontology() -> None:
    encoder = QSREProductionSchemaEncoder(cfg())
    assert torch.equal(
        encoder.layer_embedding.weight,
        torch.zeros_like(encoder.layer_embedding.weight),
    )
    assert torch.equal(
        encoder.pool_query,
        torch.zeros_like(encoder.pool_query),
    )


def test_executor_confidence_mass_survives_support_normalization() -> None:
    torch.manual_seed(21)
    executor = QSREProductionExecutor(cfg()).eval()
    g = graph_inputs()
    schema_state = torch.randn(2, 32)
    op = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0],
        applicability=1.0,
    )
    control = op.control_distribution.clone()
    control.zero_()
    control[:, 0] = 0.98
    control[:, 1] = 0.02
    cautious = QSREProductionOperatorState(
        **{**op.__dict__, "control_distribution": control}
    )
    out = executor(
        **g,
        schema_relation_state=schema_state,
        operator=cautious,
        focus_field_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert float(out["relational_probability"].sum()) <= 0.020001

    unknown = QSREProductionOperatorState(
        **{
            **op.__dict__,
            "unknown_probability": torch.full_like(op.unknown_probability, 0.99),
        }
    )
    out_unknown = executor(
        **g,
        schema_relation_state=schema_state,
        operator=unknown,
        focus_field_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert float(out_unknown["relational_probability"].sum()) <= 0.010001


def test_termination_events_cannot_reactivate_after_stop() -> None:
    torch.manual_seed(31)
    model = QSREProductionOperatorInducer(cfg()).eval()
    encoder = QSREProductionSchemaEncoder(cfg()).eval()
    q, qmask = query(batch=1, seed=32)
    s = schema(3, seed=33)
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.unknown_head.weight.zero_()
        model.stop_head.bias.fill_(30.0)
        model.unknown_head.bias.fill_(-30.0)
    out = run_operator(model, encoder, q=q, qmask=qmask, s=s, steps=5)["operator"]
    assert float(out.stop_probability[0, 0]) > 0.999
    assert float(out.stop_probability[0, 1:].abs().max()) < 1.0e-6
    assert float(out.unknown_probability.abs().max()) < 1.0e-6
    assert float(out.relation_step_mass.abs().max()) < 1.0e-6
    assert float(out.stop_probability.sum()) <= 1.000001


def test_reverse_path_execution_reaches_semantic_source() -> None:
    torch.manual_seed(31)
    executor = QSREProductionExecutor(cfg()).eval()
    g = graph_inputs()
    schema_state = torch.randn(2, 32)
    reverse = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[1, 0],
        traversal=TRAVERSAL_PATH,
        direction=DIRECTION_REVERSE,
        role=ROLE_SOURCE,
    )
    out = executor(
        **g,
        schema_relation_state=schema_state,
        operator=reverse,
        focus_field_weight=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    )
    assert int(out["path_frontier"].argmax(dim=-1).item()) == 0
    assert int(out["relational_probability"].argmax(dim=-1).item()) == 0


def test_binder_predicts_focus_without_oracle_focus_input() -> None:
    torch.manual_seed(32)
    binder = QSREProductionBinder(cfg()).eval()
    q, qmask = query(batch=1)
    fields = torch.zeros(1, 4, 5, 24)
    # Make field 2 lexically identical to the query while every distractor is
    # semantic zero, so the geometry-only focus initialization has one exact
    # deterministic optimum before the learned residual receives gradient.
    fields[0, 2, :5] = q[0, -1, :5]
    field_mask = torch.ones(1, 4, 5, dtype=torch.bool)
    s = schema(2)
    edge_index = torch.tensor([[[2, 1], [0, 3]]])
    edge_rel = torch.tensor([[0, 1]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    op = one_hot_operator(
        batch=1,
        relation_count=2,
        relation_sequence=[0],
        traversal=TRAVERSAL_PATH,
        direction=DIRECTION_FORWARD,
    )
    out = binder(
        query_hidden_states=q,
        query_token_mask=qmask,
        field_token_states=fields,
        field_token_mask=field_mask,
        field_type_id=torch.zeros(1, 4, dtype=torch.long),
        edge_index=edge_index,
        edge_relation_index=edge_rel,
        edge_valid_mask=valid,
        edge_reliability=torch.ones(1, 2),
        edge_recency=torch.ones(1, 2),
        schema=s,
        schema_relation_state=torch.randn(2, 32),
        operator=op,
    )
    assert out["focus_field_weight"].shape == (1, 4)
    assert int(out["focus_field_weight"].argmax(dim=-1).item()) == 2
    assert binder.parameter_report()["focus_is_predicted_from_query_field_late_interaction"] is True
