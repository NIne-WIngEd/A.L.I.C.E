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
                return_attention_diagnostics=True,
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
    assert report["view_reliability_causally_weights_item_contribution"] is True
    assert report["reliability_softmax_constant_cancellation"] is False
    assert report["full_slot_item_score_matrix_materialized"] is False


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


def test_latent_view_reliability_changes_item_contribution() -> None:
    torch.manual_seed(811)
    model=DynamicCompetitiveLatentPoolV3(
        DynamicLatentPoolConfig(
            semantic_dim=24,
            model_dim=24,
            slot_chunk_size=2,
            item_chunk_size=3,
            dropout=0.0,
        )
    ).eval()
    source=torch.randn(1,3,24)
    context=torch.randn(1,3,24)
    descriptor=torch.randn(1,3,24)
    available=torch.ones(1,3,dtype=torch.bool)
    query=torch.randn(1,24)
    with torch.no_grad():
        equal=model(
            source_view_summaries=source,
            contextualized_view_summaries=context,
            view_descriptor_states=descriptor,
            view_available=available,
            query_state=query,
            view_reliability=torch.ones(1,3),
            slot_count=5,
            refinement_steps=2,
        )
        suppressed=model(
            source_view_summaries=source,
            contextualized_view_summaries=context,
            view_descriptor_states=descriptor,
            view_available=available,
            query_state=query,
            view_reliability=torch.tensor([[1.0,0.0,1.0]]),
            slot_count=5,
            refinement_steps=2,
        )
    assert not torch.allclose(
        equal["pooled_state"],
        suppressed["pooled_state"],
    )


def test_latent_streamed_competition_matches_dense_one_step_reference() -> None:
    torch.manual_seed(812)
    model=DynamicCompetitiveLatentPoolV3(
        DynamicLatentPoolConfig(
            semantic_dim=24,
            model_dim=24,
            slot_chunk_size=2,
            item_chunk_size=3,
            dropout=0.0,
        )
    ).eval()
    source=torch.randn(1,4,24)
    context=torch.randn(1,4,24)
    descriptor=torch.randn(1,4,24)
    available=torch.tensor([[True,True,True,False]])
    reliability=torch.tensor([[1.0,0.7,0.2,0.0]])
    query_state=torch.randn(1,24)
    slot_count=5

    with torch.no_grad():
        actual=model(
            source_view_summaries=source,
            contextualized_view_summaries=context,
            view_descriptor_states=descriptor,
            view_available=available,
            query_state=query_state,
            view_reliability=reliability,
            slot_count=slot_count,
            refinement_steps=1,
        )

        source_p=model.source_projection(source)
        context_p=model.context_projection(context)
        desc=model.descriptor_projection(descriptor)
        query=model.query_projection(query_state)
        item=torch.stack(
            [
                source_p+desc+model.channel_embedding[0],
                context_p+desc+model.channel_embedding[1],
            ],
            dim=2,
        ).reshape(1,8,24)
        item_mask=available[:,:,None].expand(1,4,2).reshape(1,8)
        item_reliability=reliability[:,:,None].expand(1,4,2).reshape(1,8)
        coordinate=model._slot_coordinates(
            slot_count,
            device=item.device,
            dtype=item.dtype,
        )
        slot=model.output_norm(
            query[:,None,:]+model.slot_coordinate(coordinate)[None,:,:]
        )
        score=torch.einsum(
            "bsd,bid->bsi",
            model.slot_query(slot),
            model.item_key(item),
        )/(24.0**0.5)
        score=score.masked_fill(~item_mask[:,None,:],-1.0e4)
        ownership=torch.softmax(score,dim=1)
        weighted=(
            ownership
            * item_mask[:,None,:].to(ownership.dtype)
            * item_reliability[:,None,:]
        )
        attention=weighted/weighted.sum(dim=-1,keepdim=True).clamp_min(1.0e-12)
        message=torch.einsum(
            "bsi,bid->bsd",
            attention,
            model.slot_value(item),
        )
        q=query[:,None,:].expand(1,slot_count,-1)
        slot_dense=model.slot_state(
            torch.cat([message,q],dim=-1).reshape(slot_count,-1),
            slot.reshape(slot_count,-1),
        ).reshape(1,slot_count,-1)
        slot_dense=model.output_norm(slot_dense)
        pooled_score=torch.einsum(
            "bsd,bd->bs",
            slot_dense,
            model.pooled_query(query),
        )
        pooled_weight=torch.softmax(pooled_score,dim=-1)
        pooled_dense=torch.einsum(
            "bs,bsd->bd",
            pooled_weight,
            slot_dense,
        )
    assert torch.allclose(
        actual["latent_slots"],
        slot_dense,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        actual["pooled_state"],
        pooled_dense,
        atol=1e-5,
        rtol=1e-5,
    )


def test_fusion_rejects_out_of_range_view_reliability() -> None:
    model=DynamicCrossContextFusionV3(
        DynamicCrossContextFusionConfig(
            semantic_dim=24,
            model_dim=24,
            num_attention_heads=4,
            recurrent_refinement_steps=1,
            dropout=0.0,
        )
    )
    try:
        model(
            source_view_summaries=torch.randn(1,3,24),
            view_descriptor_states=torch.randn(1,3,24),
            view_available=torch.ones(1,3,dtype=torch.bool),
            query_state=torch.randn(1,24),
            view_reliability=torch.tensor([[1.0,1.2,0.5]]),
        )
    except ValueError as exc:
        assert "view_reliability" in str(exc)
        assert "[0,1]" in str(exc)
    else:
        raise AssertionError("out-of-range view reliability did not fail closed")


def test_behavioral_causal_losses_ignore_padded_candidates() -> None:
    from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
        decisive_view_causal_margin_loss,
        irrelevant_view_invariance_loss,
    )
    normal=torch.tensor([[2.0,1.0,-100.0,500.0]])
    ablated=torch.tensor([[1.0,1.0,900.0,-700.0]])
    irrelevant=torch.tensor([[2.001,0.999,800.0,-900.0]])
    valid=torch.tensor([[True,True,False,False]])
    masked_decisive=decisive_view_causal_margin_loss(
        normal,
        ablated,
        torch.tensor([0]),
        candidate_valid_mask=valid,
    )
    reference_decisive=decisive_view_causal_margin_loss(
        normal[:,:2],
        ablated[:,:2],
        torch.tensor([0]),
    )
    assert torch.allclose(
        masked_decisive,
        reference_decisive,
        atol=1e-7,
        rtol=1e-7,
    )
    masked_invariance=irrelevant_view_invariance_loss(
        normal,
        irrelevant,
        candidate_valid_mask=valid,
    )
    reference_invariance=irrelevant_view_invariance_loss(
        normal[:,:2],
        irrelevant[:,:2],
    )
    assert torch.allclose(
        masked_invariance,
        reference_invariance,
        atol=1e-7,
        rtol=1e-7,
    )


def test_latent_noncollapse_objective_does_not_create_runtime_slot_minimum() -> None:
    from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
        latent_noncollapse_loss,
    )
    slot=torch.randn(2,1,24,requires_grad=True)
    loss=latent_noncollapse_loss(slot)
    assert torch.equal(loss.detach(),torch.zeros_like(loss.detach()))
    loss.backward()
    assert slot.grad is not None


def test_latent_view_activity_causally_changes_final_pooled_state() -> None:
    torch.manual_seed(831)
    model=DynamicCompetitiveLatentPoolV3(
        DynamicLatentPoolConfig(
            semantic_dim=24,
            model_dim=24,
            slot_chunk_size=2,
            item_chunk_size=3,
            dropout=0.0,
        )
    ).eval()
    source=torch.randn(1,3,24)
    context=torch.randn(1,3,24)
    descriptor=torch.randn(1,3,24)
    available=torch.ones(1,3,dtype=torch.bool)
    query=torch.randn(1,24)
    reliability=torch.ones(1,3)
    with torch.no_grad():
        first=model(
            source_view_summaries=source,
            contextualized_view_summaries=context,
            view_descriptor_states=descriptor,
            view_available=available,
            query_state=query,
            view_reliability=reliability,
            view_activity=torch.tensor([[0.98,0.01,0.01]]),
            slot_count=5,
            refinement_steps=2,
        )
        second=model(
            source_view_summaries=source,
            contextualized_view_summaries=context,
            view_descriptor_states=descriptor,
            view_available=available,
            query_state=query,
            view_reliability=reliability,
            view_activity=torch.tensor([[0.01,0.01,0.98]]),
            slot_count=5,
            refinement_steps=2,
        )
    assert not torch.allclose(
        first["pooled_state"],
        second["pooled_state"],
    )
    report=model.parameter_report()
    assert report["query_conditioned_view_activity_supported"] is True
    assert report["view_activity_causally_weights_item_contribution"] is True


def test_semantic_operator_objective_macro_averages_query_and_schema_evidence_surfaces() -> None:
    torch.manual_seed(841)
    batch,steps,relations,tokens,schema_tokens=2,3,4,5,6
    relation_logits=torch.randn(batch,steps,relations,requires_grad=True)
    relation_targets=torch.tensor([[0,1,2],[2,1,0]])
    step_mask=torch.ones(batch,steps,dtype=torch.bool)
    factor_logits={"role":torch.randn(batch,3,requires_grad=True)}
    factor_targets={"role":torch.tensor([0,2])}
    event=torch.softmax(torch.randn(batch,steps,3),dim=-1)
    event_targets=torch.tensor([[0,0,1],[0,0,1]])
    query_evidence=torch.sigmoid(
        torch.randn(batch,steps,relations,tokens,requires_grad=True)
    )
    query_evidence.retain_grad()
    relation_schema_evidence=torch.sigmoid(
        torch.randn(batch,steps,relations,schema_tokens,requires_grad=True)
    )
    relation_schema_evidence.retain_grad()
    factor_evidence={
        "role":torch.sigmoid(torch.randn(batch,3,4,requires_grad=True))
    }
    factor_evidence["role"].retain_grad()
    step_factor_evidence={
        "role":torch.sigmoid(torch.randn(batch,steps,3,4,requires_grad=True))
    }
    step_factor_evidence["role"].retain_grad()

    result=semantic_operator_objective(
        relation_logits=relation_logits,
        relation_targets=relation_targets,
        relation_step_mask=step_mask,
        factor_logits=factor_logits,
        factor_targets=factor_targets,
        event_distribution=event,
        event_targets=event_targets,
        event_mask=step_mask,
        applicability=torch.full((batch,),0.8,requires_grad=True),
        applicability_target=torch.ones(batch),
        relation_query_evidence=query_evidence,
        query_evidence_target=torch.randint(
            0,2,(batch,steps,relations,tokens)
        ).float(),
        query_evidence_valid_mask=torch.ones(
            batch,steps,relations,tokens,dtype=torch.bool
        ),
        uncertainty=torch.full((batch,),0.2,requires_grad=True),
        uncertainty_target=torch.zeros(batch),
        correct_relation_score=torch.ones(batch),
        counterfactual_relation_score=torch.zeros(batch),
        correct_factor_score=torch.ones(batch),
        counterfactual_factor_score=torch.zeros(batch),
        relation_schema_evidence=relation_schema_evidence,
        relation_schema_evidence_target=torch.randint(
            0,2,(batch,steps,relations,schema_tokens)
        ).float(),
        relation_schema_evidence_valid_mask=torch.ones(
            batch,steps,relations,schema_tokens,dtype=torch.bool
        ),
        factor_schema_evidence=factor_evidence,
        factor_schema_evidence_target={
            "role":torch.randint(0,2,(batch,3,4)).float()
        },
        factor_schema_evidence_valid_mask={
            "role":torch.ones(batch,3,4,dtype=torch.bool)
        },
        step_factor_schema_evidence=step_factor_evidence,
        step_factor_schema_evidence_target={
            "role":torch.randint(0,2,(batch,steps,3,4)).float()
        },
        step_factor_schema_evidence_valid_mask={
            "role":torch.ones(batch,steps,3,4,dtype=torch.bool)
        },
    )
    expected=torch.stack(
        [
            result["query_evidence"],
            result["relation_schema_evidence"],
            result["factor_schema_evidence"],
            result["step_factor_schema_evidence"],
        ]
    ).mean()
    assert torch.allclose(
        result["token_evidence"],
        expected,
        atol=1e-7,
        rtol=1e-7,
    )
    result["loss"].backward()
    assert relation_schema_evidence.grad is not None
    assert float(relation_schema_evidence.grad.abs().sum()) > 0.0
    assert factor_evidence["role"].grad is not None
    assert float(factor_evidence["role"].grad.abs().sum()) > 0.0
    assert step_factor_evidence["role"].grad is not None
    assert float(step_factor_evidence["role"].grad.abs().sum()) > 0.0


def test_semantic_operator_objective_rejects_partial_schema_evidence_contract() -> None:
    try:
        semantic_operator_objective(
            relation_logits=torch.randn(1,1,2),
            relation_targets=torch.zeros(1,1,dtype=torch.long),
            relation_step_mask=torch.ones(1,1,dtype=torch.bool),
            factor_logits={"role":torch.randn(1,2)},
            factor_targets={"role":torch.zeros(1,dtype=torch.long)},
            event_distribution=torch.tensor([[[0.8,0.1,0.1]]]),
            event_targets=torch.zeros(1,1,dtype=torch.long),
            event_mask=torch.ones(1,1,dtype=torch.bool),
            applicability=torch.tensor([0.8]),
            applicability_target=torch.ones(1),
            relation_query_evidence=torch.full((1,1,2,3),0.5),
            query_evidence_target=torch.zeros(1,1,2,3),
            query_evidence_valid_mask=torch.ones(1,1,2,3,dtype=torch.bool),
            uncertainty=torch.tensor([0.2]),
            uncertainty_target=torch.zeros(1),
            correct_relation_score=torch.ones(1),
            counterfactual_relation_score=torch.zeros(1),
            correct_factor_score=torch.ones(1),
            counterfactual_factor_score=torch.zeros(1),
            relation_schema_evidence=torch.full((1,1,2,4),0.5),
        )
    except ValueError as exc:
        assert "relation schema evidence" in str(exc)
        assert "supplied together" in str(exc)
    else:
        raise AssertionError("partial schema evidence contract did not fail closed")
