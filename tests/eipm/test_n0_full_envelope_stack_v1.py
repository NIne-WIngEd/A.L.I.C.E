from __future__ import annotations

from types import SimpleNamespace

import torch

from alice_personality.n0.n0_full_envelope_stack_v1 import (
    N0FullEnvelopeStackConfig,
    N0FullEnvelopeStackV1,
)
from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
)


def schema(count: int, *, tokens: int = 4, types: int = 4, seed: int = 1) -> DynamicRelationSchema:
    torch.manual_seed(seed)
    return DynamicRelationSchema(
        token_states=torch.randn(count, 3, tokens, 24),
        token_mask=torch.ones(count, tokens, dtype=torch.bool),
        domain_type_mask=torch.ones(count, types, dtype=torch.bool),
        range_type_mask=torch.ones(count, types, dtype=torch.bool),
        symmetric=torch.zeros(count, dtype=torch.bool),
    )


def bank(count: int, seed: int) -> DynamicSemanticSchema:
    torch.manual_seed(seed)
    return DynamicSemanticSchema(
        token_states=torch.randn(count, 3, 4, 24),
        token_mask=torch.ones(count, 4, dtype=torch.bool),
    )


def factor_bundle():
    factor_schemas = {
        "role": bank(4, 11),
        "traversal": bank(3, 12),
        "direction": bank(3, 13),
        "control": bank(3, 14),
        "reliability": bank(2, 15),
        "recency": bank(2, 16),
        "temporal": bank(2, 17),
        "provenance": bank(2, 18),
    }
    factor_opcodes = {
        "role": ["ROLE_SOURCE", "ROLE_TARGET", "ROLE_SYMMETRIC", "ROLE_NONE"],
        "traversal": ["TRAVERSAL_LOCAL", "TRAVERSAL_PATH", "TRAVERSAL_AGGREGATE"],
        "direction": ["DIRECTION_FORWARD", "DIRECTION_REVERSE", "DIRECTION_BIDIRECTIONAL"],
        "control": ["CONTROL_FALLBACK", "CONTROL_RELATIONAL", "CONTROL_DEFER"],
        "reliability": ["MOD_RELIABILITY_OFF", "MOD_RELIABILITY_ON"],
        "recency": ["MOD_RECENCY_OFF", "MOD_RECENCY_ON"],
        "temporal": ["MOD_TEMPORAL_OFF", "MOD_TEMPORAL_ON"],
        "provenance": ["MOD_PROVENANCE_OFF", "MOD_PROVENANCE_ON"],
    }
    return factor_schemas, factor_opcodes


def make_inputs(batch_size: int = 2):
    batch, fields, edges = batch_size, 5, 6
    torch.manual_seed(77)
    query = torch.randn(batch, 3, 7, 24, requires_grad=True)
    field = torch.randn(batch, fields, 3, 5, 24, requires_grad=True)
    query_mask = torch.ones(batch, 7, dtype=torch.bool)
    field_mask = torch.ones(batch, fields, 5, dtype=torch.bool)
    field_valid = torch.ones(batch, fields, dtype=torch.bool)
    confidence = torch.rand(batch, fields)
    missing = torch.zeros(batch, fields)
    reliability = torch.rand(batch, fields)
    field_type = torch.tensor([[0,1,2,3,0],[1,2,3,0,1]])[:batch]
    field_metadata = torch.randn(batch, fields, 3)
    edge_index = torch.tensor([
        [[0,1],[1,2],[2,3],[3,4],[0,2],[1,4]],
        [[0,1],[1,3],[3,4],[0,2],[2,4],[1,2]],
    ])[:batch]
    edge_relation = torch.tensor([
        [0,1,2,3,4,0],
        [1,2,3,4,0,1],
    ])[:batch]
    edge_valid = torch.ones(batch, edges, dtype=torch.bool)
    edge_metadata = torch.randn(batch, edges, 4)
    edge_reliability = torch.rand(batch, edges)
    edge_recency = torch.rand(batch, edges)
    temporal = torch.ones(batch, edges)
    provenance = torch.ones(batch, edges)
    descriptors = {"type": bank(4, 31), "provenance": bank(3, 32), "temporal": bank(3, 33)}
    descriptor_indices = {
        "type": field_type,
        "provenance": torch.tensor([[0,1,2,0,1],[2,1,0,2,1]])[:batch],
        "temporal": torch.tensor([[0,1,2,0,1],[1,2,0,1,2]])[:batch],
    }
    internal_view_descriptors = torch.randn(batch, 6, 24)
    internal_view_reliability = torch.rand(batch, 6)
    return dict(
        query_hidden_states=query,
        query_token_mask=query_mask,
        field_hidden_states=field,
        field_token_mask=field_mask,
        field_valid_mask=field_valid,
        field_confidence=confidence,
        field_missing=missing,
        field_reliability=reliability,
        descriptor_banks=descriptors,
        descriptor_indices=descriptor_indices,
        field_type_index=field_type,
        field_metadata=field_metadata,
        edge_index=edge_index,
        edge_relation_index=edge_relation,
        edge_valid_mask=edge_valid,
        edge_metadata=edge_metadata,
        edge_reliability=edge_reliability,
        edge_recency=edge_recency,
        edge_temporal_match=temporal,
        edge_provenance_match=provenance,
        internal_view_descriptor_states=internal_view_descriptors,
        internal_view_reliability=internal_view_reliability,
    )


def test_full_envelope_stack_forward_and_gradient_continuity() -> None:
    model = N0FullEnvelopeStackV1(
        N0FullEnvelopeStackConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            structured_layers=1,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    )
    factors, opcodes = factor_bundle()
    inputs = make_inputs()
    candidate_hidden = torch.randn(2, 6, 3, 5, 24)
    candidate_mask = torch.ones(2, 6, 5, dtype=torch.bool)
    out = model(
        relation_schema=schema(5, seed=90),
        factor_schemas=factors,
        factor_opcodes=opcodes,
        max_reasoning_steps=3,
        graph_message_steps=2,
        fusion_refinement_steps=2,
        latent_slot_count=7,
        latent_refinement_steps=2,
        candidate_hidden_states=candidate_hidden,
        candidate_token_mask=candidate_mask,
        **inputs,
    )
    assert out["latent"]["latent_slots"].shape == (2, 7, 24)
    assert out["source_views"].shape == (2, 6, 24)
    assert out["binder"]["edge_support_weight"].shape == (2, 6)
    assert out["executor"]["relational_probability"].shape == (2, 5)
    assert out["public_judgment"]["candidate_logits"].shape == (2, 6)

    loss = (
        out["public_judgment"]["candidate_logits"].square().mean()
        + out["semantic_operator"]["relation_logits"].square().mean()
    )
    loss.backward()
    assert inputs["query_hidden_states"].grad is not None
    assert inputs["field_hidden_states"].grad is not None
    assert float(inputs["query_hidden_states"].grad.abs().sum()) > 0.0
    assert float(inputs["field_hidden_states"].grad.abs().sum()) > 0.0
    raw_gate_grads = [
        parameter.grad
        for parameter in model.raw_semantic_layer_gate.parameters()
    ]
    assert all(grad is not None for grad in raw_gate_grads)
    assert sum(float(grad.abs().sum()) for grad in raw_gate_grads if grad is not None) > 0.0
    fusion_route_grads = [
        parameter.grad
        for parameter in model.fusion.route_score.parameters()
    ]
    assert all(grad is not None for grad in fusion_route_grads)
    assert (
        sum(
            float(grad.abs().sum())
            for grad in fusion_route_grads
            if grad is not None
        )
        > 0.0
    )
    report = model.parameter_report()
    assert report["raw_semantic_view_static_layer_mean"] is False
    assert report["raw_semantic_view_content_conditioned_layer_read"] is True
    assert report["fusion_route_weight_causally_controls_latent_contribution"] is True


def test_full_envelope_stack_accepts_additional_runtime_views_and_runtime_slots() -> None:
    torch.manual_seed(88)
    model = N0FullEnvelopeStackV1(
        N0FullEnvelopeStackConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            structured_layers=1,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    factors, opcodes = factor_bundle()
    inputs = make_inputs()
    extras = 4
    with torch.no_grad():
        out = model(
            relation_schema=schema(5, seed=91),
            factor_schemas=factors,
            factor_opcodes=opcodes,
            max_reasoning_steps=5,
            graph_message_steps=3,
            fusion_refinement_steps=3,
            latent_slot_count=13,
            latent_refinement_steps=4,
            additional_source_views=torch.randn(2, extras, 24),
            additional_view_descriptor_states=torch.randn(2, extras, 24),
            additional_view_available=torch.ones(2, extras, dtype=torch.bool),
            additional_view_reliability=torch.rand(2, extras),
            **inputs,
        )
    assert out["source_views"].shape == (2, 10, 24)
    assert out["latent"]["latent_slots"].shape == (2, 13, 24)
    report = model.parameter_report()
    assert report["module_reports"]["public_judgment_probe"]["candidate_identity_parameters"] == 0
    assert report["module_reports"]["public_judgment_probe"]["candidate_count_ceiling"] is None
    assert report["runtime_relation_ceiling"] is None
    assert report["runtime_factor_ceiling"] is None
    assert report["runtime_field_ceiling"] is None
    assert report["runtime_edge_ceiling"] is None
    assert report["runtime_view_ceiling"] is None
    assert report["runtime_slot_ceiling"] is None
    assert report["runtime_reasoning_step_ceiling"] is None
    assert report["exact_structural_sparsity_boundary"] == "FullEnvelopeQSREBinderV1"


def _summary_operator(
    *,
    relation_distribution: torch.Tensor,
    relation_step_mass: torch.Tensor,
    stop_probability: torch.Tensor,
    truncation_probability: torch.Tensor,
) -> SimpleNamespace:
    completed_program=stop_probability[:,1:].sum(dim=1).clamp(0.0,1.0)
    reverse=torch.flip(
        torch.cumsum(torch.flip(stop_probability,dims=[1]),dim=1),
        dims=[1],
    )
    completed_step=(reverse-stop_probability).clamp(0.0,1.0)
    denominator=torch.where(
        completed_program>0.0,
        completed_program,
        torch.ones_like(completed_program),
    )
    completed_weight=torch.where(
        completed_program[:,None]>0.0,
        completed_step/denominator[:,None],
        torch.zeros_like(completed_step),
    )
    return SimpleNamespace(
        relation_distribution=relation_distribution,
        relation_step_mass=relation_step_mass,
        stop_probability=stop_probability,
        unknown_probability=torch.zeros_like(relation_step_mass),
        truncation_probability=truncation_probability,
        applicability=torch.ones(relation_step_mass.size(0)),
        control_distribution=torch.tensor(
            [[0.0,1.0,0.0]],
            dtype=relation_step_mass.dtype,
        ).expand(relation_step_mass.size(0),-1).clone(),
        completed_relational_program_mass=lambda: completed_program,
        completed_relational_step_weight=lambda: completed_weight,
    )


def test_stack_semantic_activity_uses_exact_completed_program_mass_not_expected_step_count() -> None:
    operator=_summary_operator(
        relation_distribution=torch.tensor(
            [[[1.0,0.0],[0.0,1.0],[1.0,0.0]]]
        ),
        relation_step_mass=torch.tensor([[0.2,0.2,0.0]]),
        stop_probability=torch.tensor([[0.8,0.0,0.2]]),
        truncation_probability=torch.zeros(1),
    )
    relation_mass,activity=N0FullEnvelopeStackV1._relation_program_summary(
        operator
    )
    assert torch.allclose(activity,torch.tensor([0.2]),atol=1e-6)
    assert torch.allclose(
        relation_mass,
        torch.tensor([[0.5,0.5]]),
        atol=1e-6,
    )


def test_truncated_program_cannot_activate_global_relational_views() -> None:
    """Residual CONTINUE survival is incomplete execution, not completion."""
    operator=_summary_operator(
        relation_distribution=torch.tensor(
            [[[1.0,0.0],[0.0,1.0],[1.0,0.0]]]
        ),
        relation_step_mass=torch.tensor([[1.0,1.0,1.0]]),
        stop_probability=torch.zeros(1,3),
        truncation_probability=torch.ones(1),
    )
    _,activity=N0FullEnvelopeStackV1._relation_program_summary(operator)
    assert torch.equal(activity,torch.zeros_like(activity))


def test_full_stack_disables_unavailable_graph_and_executor_views() -> None:
    reliability=torch.tensor(
        [
            [1.0,0.9,0.8,0.7,0.6,0.5],
            [1.0,0.9,0.8,0.7,0.6,0.5],
        ]
    )
    available,effective=N0FullEnvelopeStackV1._internal_view_gates(
        internal_view_reliability=reliability,
        graph_support_activity=torch.tensor([0.0,0.4]),
        execution_confidence=torch.tensor([0.0,0.25]),
    )
    assert available[0].tolist()==[True,True,True,False,False,False]
    assert available[1].tolist()==[True,True,True,True,True,True]
    assert torch.equal(
        effective[0,3:],
        torch.zeros_like(effective[0,3:]),
    )
    assert torch.allclose(
        effective[1,3:],
        torch.tensor([0.28,0.24,0.125]),
        atol=1e-7,
    )


def test_graph_views_require_exact_binder_support_before_final_fusion() -> None:
    graph_activity=torch.tensor([0.9,0.7,0.4])
    binder_support=torch.tensor([0.0,0.5,1.0])
    supported=N0FullEnvelopeStackV1._supported_graph_view_activity(
        graph_support_activity=graph_activity,
        binder_support_available=binder_support,
    )
    assert torch.allclose(
        supported,
        torch.tensor([0.0,0.35,0.4]),
        atol=1e-7,
    )
    availability,reliability=N0FullEnvelopeStackV1._internal_view_gates(
        internal_view_reliability=torch.ones(3,6),
        graph_support_activity=supported,
        execution_confidence=torch.tensor([0.0,0.2,0.3]),
    )
    assert availability[0].tolist()==[True,True,True,False,False,False]
    assert torch.equal(
        reliability[0,3:],
        torch.zeros_like(reliability[0,3:]),
    )


def test_full_stack_pregraph_type_gate_matches_exact_binder_compatibility() -> None:
    torch.manual_seed(292)
    model=N0FullEnvelopeStackV1(
        N0FullEnvelopeStackConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            structured_layers=1,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    factors,opcodes=factor_bundle()
    inputs=make_inputs(batch_size=1)
    relation=DynamicRelationSchema(
        token_states=torch.randn(5,3,4,24),
        token_mask=torch.ones(5,4,dtype=torch.bool),
        domain_type_mask=torch.tensor(
            [
                [True,False,False,False],
                [False,True,False,False],
                [False,False,True,False],
                [False,False,False,True],
                [True,False,False,False],
            ],
            dtype=torch.bool,
        ),
        range_type_mask=torch.tensor(
            [
                [False,True,False,False],
                [False,False,True,False],
                [False,False,False,True],
                [True,False,False,False],
                [False,False,True,False],
            ],
            dtype=torch.bool,
        ),
        symmetric=torch.tensor([False,False,False,False,False]),
    )
    with torch.no_grad():
        out=model(
            relation_schema=relation,
            factor_schemas=factors,
            factor_opcodes=opcodes,
            max_reasoning_steps=3,
            graph_message_steps=2,
            fusion_refinement_steps=2,
            latent_slot_count=5,
            latent_refinement_steps=2,
            **inputs,
        )
    assert torch.equal(
        out["pregraph_type_compatible"],
        out["binder"]["type_compatible"],
    )
    assert bool((~out["pregraph_type_compatible"]).any())
    report=model.parameter_report()
    assert report["pre_binder_graph_exact_type_schema_gated"] is True


def test_additional_runtime_view_order_has_no_hidden_identity_axis() -> None:
    torch.manual_seed(294)
    model=N0FullEnvelopeStackV1(
        N0FullEnvelopeStackConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            structured_layers=1,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    factors,opcodes=factor_bundle()
    inputs=make_inputs(batch_size=1)
    extras=torch.randn(1,4,24)
    descriptors=torch.randn(1,4,24)
    available=torch.tensor([[True,True,False,True]],dtype=torch.bool)
    reliability=torch.tensor([[0.9,0.3,0.8,0.6]])
    candidate_hidden=torch.randn(1,5,3,4,24)
    candidate_mask=torch.ones(1,5,4,dtype=torch.bool)
    candidate_valid=torch.tensor([[True,True,True,False,True]],dtype=torch.bool)
    perm=torch.tensor([3,0,2,1])

    common=dict(
        relation_schema=schema(5,seed=295),
        factor_schemas=factors,
        factor_opcodes=opcodes,
        max_reasoning_steps=4,
        graph_message_steps=2,
        fusion_refinement_steps=2,
        latent_slot_count=6,
        latent_refinement_steps=2,
        candidate_hidden_states=candidate_hidden,
        candidate_token_mask=candidate_mask,
        candidate_valid_mask=candidate_valid,
        **inputs,
    )
    with torch.no_grad():
        original=model(
            additional_source_views=extras,
            additional_view_descriptor_states=descriptors,
            additional_view_available=available,
            additional_view_reliability=reliability,
            **common,
        )
        permuted=model(
            additional_source_views=extras[:,perm],
            additional_view_descriptor_states=descriptors[:,perm],
            additional_view_available=available[:,perm],
            additional_view_reliability=reliability[:,perm],
            **common,
        )

    assert torch.allclose(
        original["fusion"]["fused_state"],
        permuted["fusion"]["fused_state"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.allclose(
        original["latent"]["pooled_state"],
        permuted["latent"]["pooled_state"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.allclose(
        original["public_judgment"]["candidate_logits"],
        permuted["public_judgment"]["candidate_logits"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
