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
