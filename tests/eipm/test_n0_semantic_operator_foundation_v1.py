from __future__ import annotations

import torch

from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
    SchemaConditionedSemanticOperator,
    SemanticOperatorFoundationConfig,
)


def config() -> SemanticOperatorFoundationConfig:
    return SemanticOperatorFoundationConfig(
        semantic_dim=24,
        model_dim=24,
        num_hidden_states=3,
        dropout=0.0,
    )


def hidden(batch: int = 2, tokens: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    value = torch.randn(batch, 3, tokens, 24)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    mask[:, -1] = False
    return value, mask


def schema(count: int, tokens: int = 5, types: int = 6) -> DynamicRelationSchema:
    torch.manual_seed(100 + count)
    states = torch.randn(count, 3, tokens, 24)
    mask = torch.ones(count, tokens, dtype=torch.bool)
    domain = torch.zeros(count, types, dtype=torch.bool)
    range_mask = torch.zeros_like(domain)
    for i in range(count):
        domain[i, i % types] = True
        range_mask[i, (i + 1) % types] = True
    symmetric = torch.zeros(count, dtype=torch.bool)
    return DynamicRelationSchema(
        token_states=states,
        token_mask=mask,
        domain_type_mask=domain,
        range_type_mask=range_mask,
        symmetric=symmetric,
    )


def factor(count: int, seed: int) -> DynamicSemanticSchema:
    torch.manual_seed(seed)
    return DynamicSemanticSchema(
        token_states=torch.randn(count, 3, 4, 24),
        token_mask=torch.ones(count, 4, dtype=torch.bool),
    )


def test_runtime_relation_and_factor_cardinality_have_no_parameter_axis() -> None:
    model = SchemaConditionedSemanticOperator(config())
    q, qm = hidden()
    before = model.parameter_report()["total_parameters"]

    small = model(
        query_hidden_states=q,
        query_token_mask=qm,
        relation_schema=schema(3),
        factor_schemas={"role": factor(2, 1), "control": factor(3, 2)},
        max_steps=2,
    )
    large = model(
        query_hidden_states=q,
        query_token_mask=qm,
        relation_schema=schema(11, types=13),
        factor_schemas={"role": factor(7, 3), "new_runtime_factor": factor(5, 4)},
        max_steps=5,
    )

    assert small["operator"].relation_distribution.shape == (2, 2, 3)
    assert large["operator"].relation_distribution.shape == (2, 5, 11)
    assert small["operator"].factor_distributions["role"].shape == (2, 2)
    assert large["operator"].factor_distributions["role"].shape == (2, 7)
    assert large["operator"].factor_distributions["new_runtime_factor"].shape == (2, 5)
    assert model.parameter_report()["total_parameters"] == before

    report = model.parameter_report()
    assert report["relation_identity_parameters"] == 0
    assert report["factor_identity_parameters"] == 0
    assert report["candidate_count_dependent_parameters"] == 0
    assert report["runtime_step_count_dependent_parameters"] == 0
    assert report["type_vocabulary_dependent_parameters"] == 0
    assert report["relation_count_ceiling"] is None
    assert report["factor_count_ceiling"] is None
    assert report["runtime_step_count_ceiling"] is None
    assert report["type_vocabulary_ceiling"] is None


def test_relation_candidate_permutation_is_equivariant() -> None:
    torch.manual_seed(13)
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden()
    original = schema(5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = DynamicRelationSchema(
        token_states=original.token_states[permutation],
        token_mask=original.token_mask[permutation],
        domain_type_mask=original.domain_type_mask[permutation],
        range_type_mask=original.range_type_mask[permutation],
        symmetric=original.symmetric[permutation],
    )

    with torch.no_grad():
        a = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=original,
            factor_schemas={},
            max_steps=3,
        )["operator"].relation_distribution
        b = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=permuted,
            factor_schemas={},
            max_steps=3,
        )["operator"].relation_distribution

    assert torch.allclose(b, a[..., permutation], atol=1e-5, rtol=1e-5)


def test_factor_candidate_permutation_is_equivariant() -> None:
    torch.manual_seed(17)
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden()
    bank = factor(6, 23)
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    permuted = DynamicSemanticSchema(
        token_states=bank.token_states[permutation],
        token_mask=bank.token_mask[permutation],
    )

    with torch.no_grad():
        a = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={"runtime_factor": bank},
            max_steps=2,
        )["operator"].factor_distributions["runtime_factor"]
        b = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={"runtime_factor": permuted},
            max_steps=2,
        )["operator"].factor_distributions["runtime_factor"]

    assert torch.allclose(b, a[:, permutation], atol=1e-5, rtol=1e-5)


def test_relation_semantics_remain_continuous_and_query_coverage_is_monotonic() -> None:
    torch.manual_seed(19)
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden()
    with torch.no_grad():
        operator = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(7),
            factor_schemas={"role": factor(4, 29)},
            max_steps=4,
        )["operator"]

    probability = operator.relation_distribution
    assert torch.allclose(
        probability.sum(dim=-1),
        torch.ones_like(probability.sum(dim=-1)),
        atol=1e-6,
    )
    assert bool((probability > 0).all())
    assert operator.event_distribution.shape[-1] == 3
    assert torch.allclose(
        operator.event_distribution.sum(dim=-1),
        torch.ones_like(operator.event_distribution.sum(dim=-1)),
        atol=1e-6,
    )
    assert bool(
        (
            operator.query_coverage[:, 1:]
            >= operator.query_coverage[:, :-1] - 1e-7
        ).all()
    )


def test_step_count_changes_runtime_axis_not_parameters() -> None:
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden()
    count = model.parameter_report()["total_parameters"]
    with torch.no_grad():
        one = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={},
            max_steps=1,
        )["operator"]
        six = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={},
            max_steps=6,
        )["operator"]
    assert one.relation_distribution.shape[1] == 1
    assert six.relation_distribution.shape[1] == 6
    assert model.parameter_report()["total_parameters"] == count


def test_schema_validation_fails_closed() -> None:
    good = schema(3)
    bad = DynamicRelationSchema(
        token_states=good.token_states,
        token_mask=torch.zeros_like(good.token_mask),
        domain_type_mask=good.domain_type_mask,
        range_type_mask=good.range_type_mask,
        symmetric=good.symmetric,
    )
    try:
        bad.validate(num_hidden_states=3, semantic_dim=24)
    except ValueError as exc:
        assert "content token" in str(exc)
    else:
        raise AssertionError("empty semantic schema candidate did not fail closed")


def test_semantic_backbone_gradient_is_not_detached_by_operator() -> None:
    model = SchemaConditionedSemanticOperator(config())
    q, qm = hidden(batch=1)
    q.requires_grad_(True)
    out = model(
        query_hidden_states=q,
        query_token_mask=qm,
        relation_schema=schema(3),
        factor_schemas={"role": factor(3, 31)},
        max_steps=2,
    )
    loss = out["relation_logits"].sum() + out["factor_logits"]["role"].sum()
    loss.backward()
    assert q.grad is not None
    assert bool(torch.isfinite(q.grad).all())
    assert float(q.grad.abs().sum()) > 0.0


def test_per_example_candidate_masks_enable_variable_runtime_subsets_in_one_batch() -> None:
    torch.manual_seed(37)
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden(batch=2)
    relation_bank = schema(6)
    role_bank = factor(4, 41)
    relation_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [False, True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    role_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )
    with torch.no_grad():
        out = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=relation_bank,
            factor_schemas={"role": role_bank},
            max_steps=3,
            relation_candidate_mask=relation_mask,
            factor_candidate_masks={"role": role_mask},
        )
    relation = out["operator"].relation_distribution
    role = out["operator"].factor_distributions["role"]
    step_role = out["operator"].step_factor_distributions["role"]
    assert torch.equal(
        relation.masked_select(~relation_mask[:, None, :]),
        torch.zeros_like(relation.masked_select(~relation_mask[:, None, :])),
    )
    assert torch.equal(
        role.masked_select(~role_mask),
        torch.zeros_like(role.masked_select(~role_mask)),
    )
    expanded_role_mask = role_mask[:, None, :].expand_as(step_role)
    assert torch.equal(
        step_role.masked_select(~expanded_role_mask),
        torch.zeros_like(step_role.masked_select(~expanded_role_mask)),
    )
    assert torch.allclose(
        relation.sum(dim=-1),
        torch.ones_like(relation.sum(dim=-1)),
        atol=1e-6,
    )
    assert model.parameter_report()["per_example_candidate_subset_supported"] is True


def test_single_active_candidate_mask_has_zero_relation_entropy_not_fake_confidence_margin_failure() -> None:
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden(batch=1)
    mask = torch.tensor([[False, True, False]], dtype=torch.bool)
    with torch.no_grad():
        out = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(3),
            factor_schemas={},
            max_steps=2,
            relation_candidate_mask=mask,
        )
    probability = out["operator"].relation_distribution
    assert torch.equal(
        probability[..., 0],
        torch.zeros_like(probability[..., 0]),
    )
    assert torch.equal(
        probability[..., 2],
        torch.zeros_like(probability[..., 2]),
    )
    assert torch.allclose(
        probability[..., 1],
        torch.ones_like(probability[..., 1]),
        atol=1e-6,
    )


def test_residual_program_survival_is_exposed_as_truncation_and_uncertainty() -> None:
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden(batch=1)
    with torch.no_grad():
        operator = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={"role": factor(4, 43)},
            max_steps=1,
        )["operator"]
    assert operator.truncation_probability.shape == (1,)
    assert bool((operator.truncation_probability >= 0).all())
    assert bool((operator.truncation_probability <= 1).all())
    assert model.parameter_report()["runtime_step_truncation_exposed"] is True
    assert model.parameter_report()["silent_program_truncation_forbidden"] is True


def test_step_conditioned_factor_distributions_exist_for_every_reasoning_slot() -> None:
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden(batch=2)
    with torch.no_grad():
        operator = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(5),
            factor_schemas={
                "direction": factor(3, 47),
                "reliability_modifier": factor(2, 48),
            },
            max_steps=4,
        )["operator"]
    assert operator.step_factor_distributions["direction"].shape == (2,4,3)
    assert operator.step_factor_distributions["reliability_modifier"].shape == (2,4,2)
    assert torch.allclose(
        operator.step_factor_distributions["direction"].sum(dim=-1),
        torch.ones(2,4),
        atol=1e-6,
    )


def test_masked_runtime_relation_padding_does_not_change_operator_uncertainty() -> None:
    torch.manual_seed(113)
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden(batch=1)
    base = schema(3, types=6)
    torch.manual_seed(114)
    extra_states = torch.randn(2, 3, 5, 24)
    extended = DynamicRelationSchema(
        token_states=torch.cat([base.token_states, extra_states], dim=0),
        token_mask=torch.cat(
            [base.token_mask, torch.ones(2, 5, dtype=torch.bool)],
            dim=0,
        ),
        domain_type_mask=torch.cat(
            [base.domain_type_mask, torch.ones(2, 6, dtype=torch.bool)],
            dim=0,
        ),
        range_type_mask=torch.cat(
            [base.range_type_mask, torch.ones(2, 6, dtype=torch.bool)],
            dim=0,
        ),
        symmetric=torch.cat(
            [base.symmetric, torch.zeros(2, dtype=torch.bool)],
            dim=0,
        ),
    )
    factors = {"role": factor(4, 115)}
    with torch.no_grad():
        a = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=base,
            factor_schemas=factors,
            max_steps=3,
        )["operator"]
        b = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=extended,
            factor_schemas=factors,
            max_steps=3,
            relation_candidate_mask=torch.tensor(
                [[True, True, True, False, False]],
                dtype=torch.bool,
            ),
        )["operator"]
    assert torch.allclose(
        a.relation_distribution,
        b.relation_distribution[..., :3],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a.uncertainty,
        b.uncertainty,
        atol=1e-6,
        rtol=1e-6,
    )
    assert model.parameter_report()["masked_candidate_uncertainty_normalization"] is True


def test_runtime_factor_bank_order_is_semantically_permutation_invariant() -> None:
    torch.manual_seed(121)
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden(batch=2)
    alpha = factor(3, 122)
    beta = factor(5, 123)
    with torch.no_grad():
        a = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={"alpha": alpha, "beta": beta},
            max_steps=3,
        )
        b = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={"beta": beta, "alpha": alpha},
            max_steps=3,
        )
    assert torch.allclose(
        a["operator"].continuous_state,
        b["operator"].continuous_state,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a["operator"].factor_distributions["alpha"],
        b["operator"].factor_distributions["alpha"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a["operator"].factor_distributions["beta"],
        b["operator"].factor_distributions["beta"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert model.parameter_report()["runtime_factor_bank_set_aggregation"] is True


def test_semantic_only_factor_bank_changes_continuous_operator_context() -> None:
    torch.manual_seed(124)
    model = SchemaConditionedSemanticOperator(config()).eval()
    q, qm = hidden(batch=1)
    structural = factor(4, 125)
    semantic_a = factor(3, 126)
    semantic_b = factor(3, 127)
    with torch.no_grad():
        a = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={
                "role": structural,
                "open_semantic_factor": semantic_a,
            },
            max_steps=3,
        )
        b = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=schema(4),
            factor_schemas={
                "role": structural,
                "open_semantic_factor": semantic_b,
            },
            max_steps=3,
        )
    assert not torch.allclose(
        a["operator"].continuous_state,
        b["operator"].continuous_state,
    )
    assert a["factor_context_state"].shape == (1, 24)
    assert a["step_factor_context_states"].shape == (1, 3, 24)
    report = model.parameter_report()
    assert report["semantic_factor_context_in_continuous_state"] is True
    assert report["semantic_factor_bank_count_ceiling"] is None
