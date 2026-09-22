from __future__ import annotations

import torch

from alice_personality.n0.dynamic_competitive_latent_pool_v3 import (
    DynamicCompetitiveLatentPoolV3,
    DynamicLatentPoolConfig,
)
from alice_personality.n0.dynamic_cross_context_fusion_v3 import (
    DynamicCrossContextFusionConfig,
    DynamicCrossContextFusionV3,
)
from alice_personality.n0.dynamic_evidence_view_v2 import (
    DynamicEvidenceViewConfig,
    DynamicEvidenceViewV2,
)
from alice_personality.n0.semantic_operator_objectives_v1 import (
    semantic_operator_objective,
)


def test_dynamic_evidence_view_uses_multilayer_tokens_and_has_no_relation_axis_params() -> None:
    torch.manual_seed(101)
    model = DynamicEvidenceViewV2(
        DynamicEvidenceViewConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            dropout=0.0,
        )
    ).eval()

    batch, fields, relations = 2, 5, 7
    query = torch.randn(batch, 3, 6, 24)
    query_mask = torch.ones(batch, 6, dtype=torch.bool)
    field = torch.randn(batch, fields, 3, 4, 24)
    field_mask = torch.ones(batch, fields, 4, dtype=torch.bool)
    structured = torch.randn(batch, fields, 24)
    valid = torch.ones(batch, fields, dtype=torch.bool)
    confidence = torch.rand(batch, fields)
    missing = torch.zeros(batch, fields)
    reliability = torch.rand(batch, fields)
    relation_state = torch.randn(batch, relations, 24)
    relation_mass = torch.softmax(torch.randn(batch, relations), dim=-1)
    operator = torch.randn(batch, 24)

    with torch.no_grad():
        out = model(
            query_hidden_states=query,
            query_token_mask=query_mask,
            field_hidden_states=field,
            field_token_mask=field_mask,
            structured_field_state=structured,
            field_valid_mask=valid,
            field_confidence=confidence,
            field_missing=missing,
            field_reliability=reliability,
            relation_schema_state=relation_state,
            relation_mass=relation_mass,
            semantic_activity=torch.ones(batch),
            operator_state=operator,
        )
    assert out["evidence_field_state"].shape == (batch, fields, 24)
    assert out["field_weight"].shape == (batch, fields)
    assert torch.allclose(
        out["field_weight"].sum(dim=-1),
        torch.ones(batch),
        atol=1e-6,
    )
    report = model.parameter_report()
    assert report["relation_identity_parameters"] == 0
    assert report["relation_count_dependent_parameters"] == 0
    assert report["pooled_only_query_conditioning"] is False
    assert report["exact_structural_sparsity"] is False


def test_dynamic_fusion_is_view_permutation_equivariant_and_source_exact() -> None:
    torch.manual_seed(202)
    model = DynamicCrossContextFusionV3(
        DynamicCrossContextFusionConfig(
            semantic_dim=24,
            model_dim=24,
            num_attention_heads=4,
            recurrent_refinement_steps=2,
            dropout=0.0,
        )
    ).eval()

    batch, views = 2, 6
    source = torch.randn(batch, views, 24)
    descriptor = torch.randn(batch, views, 24)
    available = torch.ones(batch, views, dtype=torch.bool)
    available[0, -1] = False
    query = torch.randn(batch, 24)
    reliability = torch.rand(batch, views)
    permutation = torch.tensor([4, 0, 5, 1, 3, 2])

    with torch.no_grad():
        a = model(
            source_view_summaries=source,
            view_descriptor_states=descriptor,
            view_available=available,
            query_state=query,
            view_reliability=reliability,
        )
        b = model(
            source_view_summaries=source[:, permutation],
            view_descriptor_states=descriptor[:, permutation],
            view_available=available[:, permutation],
            query_state=query,
            view_reliability=reliability[:, permutation],
        )

    assert torch.equal(a["source_view_summaries"], source)
    assert torch.equal(
        b["source_view_summaries"],
        source[:, permutation],
    )
    assert torch.allclose(
        b["contextualized_view_summaries"],
        a["contextualized_view_summaries"][:, permutation],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        b["view_weight"],
        a["view_weight"][:, permutation],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        b["fused_state"],
        a["fused_state"],
        atol=1e-5,
        rtol=1e-5,
    )
    report = model.parameter_report()
    assert report["view_identity_parameters"] == 0
    assert report["view_count_dependent_parameters"] == 0
    assert report["view_count_ceiling"] is None
    assert report["full_view_pair_score_matrix_materialized"] is False
    assert report["exact_dense_view_attention_semantics"] is True


def test_dynamic_latent_pool_supports_runtime_slot_and_view_counts_without_param_growth() -> None:
    torch.manual_seed(303)
    model = DynamicCompetitiveLatentPoolV3(
        DynamicLatentPoolConfig(
            semantic_dim=24,
            model_dim=24,
            dropout=0.0,
        )
    ).eval()
    before = model.parameter_report()["total_parameters"]

    def run(views: int, slots: int):
        source = torch.randn(2, views, 24)
        context = torch.randn(2, views, 24)
        descriptor = torch.randn(2, views, 24)
        available = torch.ones(2, views, dtype=torch.bool)
        query = torch.randn(2, 24)
        reliability = torch.rand(2, views)
        with torch.no_grad():
            return model(
                source_view_summaries=source,
                contextualized_view_summaries=context,
                view_descriptor_states=descriptor,
                view_available=available,
                query_state=query,
                view_reliability=reliability,
                slot_count=slots,
                refinement_steps=3,
            )

    small = run(3, 4)
    large = run(9, 17)
    assert small["latent_slots"].shape == (2, 4, 24)
    assert large["latent_slots"].shape == (2, 17, 24)
    assert small["view_attention_mass"].shape == (2, 4, 3)
    assert large["view_attention_mass"].shape == (2, 17, 9)
    assert model.parameter_report()["total_parameters"] == before

    report = model.parameter_report()
    assert report["slot_identity_parameters"] == 0
    assert report["view_identity_parameters"] == 0
    assert report["slot_count_dependent_parameters"] == 0
    assert report["view_count_dependent_parameters"] == 0
    assert report["slot_count_ceiling"] is None
    assert report["view_count_ceiling"] is None
    assert report["competitive_item_ownership"] is True


def test_dynamic_latent_pool_is_view_permutation_invariant_at_pooled_output() -> None:
    torch.manual_seed(404)
    model = DynamicCompetitiveLatentPoolV3(
        DynamicLatentPoolConfig(
            semantic_dim=24,
            model_dim=24,
            dropout=0.0,
        )
    ).eval()

    source = torch.randn(2, 5, 24)
    context = torch.randn(2, 5, 24)
    descriptor = torch.randn(2, 5, 24)
    available = torch.ones(2, 5, dtype=torch.bool)
    query = torch.randn(2, 24)
    reliability = torch.rand(2, 5)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    with torch.no_grad():
        a = model(
            source_view_summaries=source,
            contextualized_view_summaries=context,
            view_descriptor_states=descriptor,
            view_available=available,
            query_state=query,
            view_reliability=reliability,
            slot_count=7,
            refinement_steps=2,
        )
        b = model(
            source_view_summaries=source[:, permutation],
            contextualized_view_summaries=context[:, permutation],
            view_descriptor_states=descriptor[:, permutation],
            view_available=available[:, permutation],
            query_state=query,
            view_reliability=reliability[:, permutation],
            slot_count=7,
            refinement_steps=2,
        )

    assert torch.allclose(
        a["latent_slots"],
        b["latent_slots"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        a["pooled_state"],
        b["pooled_state"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_semantic_operator_objective_is_behavioral_and_finite() -> None:
    torch.manual_seed(505)
    batch, steps, relations, tokens = 3, 4, 6, 8
    relation_logits = torch.randn(batch, steps, relations, requires_grad=True)
    relation_targets = torch.randint(0, relations, (batch, steps))
    relation_mask = torch.ones(batch, steps, dtype=torch.bool)

    factor_logits = {
        "role": torch.randn(batch, 4, requires_grad=True),
        "control": torch.randn(batch, 3, requires_grad=True),
    }
    factor_targets = {
        "role": torch.randint(0, 4, (batch,)),
        "control": torch.randint(0, 3, (batch,)),
    }
    step_factor_logits = {
        "direction": torch.randn(batch, steps, 3, requires_grad=True),
        "reliability_modifier": torch.randn(batch, steps, 2, requires_grad=True),
    }
    step_factor_targets = {
        "direction": torch.randint(0, 3, (batch, steps)),
        "reliability_modifier": torch.randint(0, 2, (batch, steps)),
    }
    event = torch.softmax(torch.randn(batch, steps, 3), dim=-1)
    event_targets = torch.randint(0, 3, (batch, steps))
    event_mask = torch.ones(batch, steps, dtype=torch.bool)
    applicability = torch.sigmoid(torch.randn(batch, requires_grad=True))
    applicability_target = torch.randint(0, 2, (batch,)).float()

    query_evidence = torch.sigmoid(
        torch.randn(batch, steps, relations, tokens, requires_grad=True)
    )
    query_target = torch.randint(
        0, 2, (batch, steps, relations, tokens)
    ).float()
    query_valid = torch.ones(
        batch, steps, relations, tokens, dtype=torch.bool
    )
    uncertainty = torch.sigmoid(torch.randn(batch, requires_grad=True))
    uncertainty_target = torch.rand(batch)

    result = semantic_operator_objective(
        relation_logits=relation_logits,
        relation_targets=relation_targets,
        relation_step_mask=relation_mask,
        factor_logits=factor_logits,
        factor_targets=factor_targets,
        event_distribution=event,
        event_targets=event_targets,
        event_mask=event_mask,
        applicability=applicability,
        applicability_target=applicability_target,
        relation_query_evidence=query_evidence,
        query_evidence_target=query_target,
        query_evidence_valid_mask=query_valid,
        uncertainty=uncertainty,
        uncertainty_target=uncertainty_target,
        correct_relation_score=torch.ones(batch),
        counterfactual_relation_score=torch.zeros(batch),
        correct_factor_score=torch.ones(batch),
        counterfactual_factor_score=torch.zeros(batch),
        step_factor_logits=step_factor_logits,
        step_factor_targets=step_factor_targets,
        step_factor_mask=relation_mask,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert relation_logits.grad is not None
    assert factor_logits["role"].grad is not None
    assert step_factor_logits["direction"].grad is not None
    assert torch.isfinite(result["step_factor_semantics"])


def test_latent_slot_seed_coordinates_are_count_stable() -> None:
    small = DynamicCompetitiveLatentPoolV3._slot_coordinates(
        5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    large = DynamicCompetitiveLatentPoolV3._slot_coordinates(
        17,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(small, large[:5])
    model = DynamicCompetitiveLatentPoolV3(
        DynamicLatentPoolConfig(
            semantic_dim=24,
            model_dim=24,
            dropout=0.0,
        )
    )
    assert model.parameter_report()["slot_seed_coordinates_count_stable"] is True


def test_public_judgment_probe_supports_padded_candidate_subsets() -> None:
    from alice_personality.n0.public_judgment_probe_v1 import (
        PublicJudgmentProbeConfig,
        PublicJudgmentProbeV1,
    )
    from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
        public_judgment_loss,
    )

    torch.manual_seed(707)
    probe=PublicJudgmentProbeV1(
        PublicJudgmentProbeConfig(
            semantic_dim=24,
            latent_dim=24,
            model_dim=24,
            num_hidden_states=3,
        )
    ).eval()
    hidden=torch.randn(2,5,3,4,24)
    token_mask=torch.ones(2,5,4,dtype=torch.bool)
    valid=torch.tensor(
        [[True,True,False,False,False],
         [True,True,True,True,False]],
        dtype=torch.bool,
    )
    token_mask[~valid]=False
    with torch.no_grad():
        out=probe(
            pooled_state=torch.randn(2,24),
            candidate_hidden_states=hidden,
            candidate_token_mask=token_mask,
            candidate_valid_mask=valid,
        )
    assert torch.equal(
        out["candidate_logits"].masked_select(~valid),
        torch.full_like(
            out["candidate_logits"].masked_select(~valid),
            -1.0e4,
        ),
    )
    loss=public_judgment_loss(
        out["candidate_logits"],
        torch.tensor([1,3]),
        candidate_valid_mask=valid,
    )
    assert torch.isfinite(loss)
    assert probe.parameter_report()["padded_candidate_batching_supported"] is True


def test_evidence_view_inactive_relation_context_is_schema_invariant() -> None:
    torch.manual_seed(181)
    model = DynamicEvidenceViewV2(
        DynamicEvidenceViewConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            dropout=0.0,
        )
    ).eval()
    batch,fields,relations=1,4,3
    common=dict(
        query_hidden_states=torch.randn(batch,3,6,24),
        query_token_mask=torch.ones(batch,6,dtype=torch.bool),
        field_hidden_states=torch.randn(batch,fields,3,5,24),
        field_token_mask=torch.ones(batch,fields,5,dtype=torch.bool),
        structured_field_state=torch.randn(batch,fields,24),
        field_valid_mask=torch.ones(batch,fields,dtype=torch.bool),
        field_confidence=torch.rand(batch,fields),
        field_missing=torch.zeros(batch,fields),
        field_reliability=torch.rand(batch,fields),
        relation_mass=torch.softmax(torch.randn(batch,relations),dim=-1),
        semantic_activity=torch.zeros(batch),
        operator_state=torch.randn(batch,24),
    )
    with torch.no_grad():
        a=model(
            relation_schema_state=torch.randn(batch,relations,24),
            **common,
        )
        b=model(
            relation_schema_state=torch.randn(batch,relations,24)*9.0,
            **common,
        )
    assert torch.allclose(
        a["evidence_summary"],
        b["evidence_summary"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        a["field_weight"],
        b["field_weight"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert model.parameter_report()["soft_relation_context_activity_gate"] is True
