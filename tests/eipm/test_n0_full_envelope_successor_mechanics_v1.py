from __future__ import annotations

import torch

from alice_personality.n0.dynamic_schema_evidence_graph_v1 import (
    DynamicSchemaEvidenceGraphConfig,
    DynamicSchemaEvidenceGraphV1,
)
from alice_personality.n0.dynamic_structured_state_v2 import (
    DynamicStructuredStateConfig,
    DynamicStructuredStateV2,
)
from alice_personality.n0.qsre_full_envelope_binder_v1 import (
    FullEnvelopeBinderConfig,
    FullEnvelopeQSREBinderV1,
)
from alice_personality.n0.qsre_full_envelope_executor_v1 import (
    FullEnvelopeExecutorConfig,
    FullEnvelopeQSREExecutorV1,
)
from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
    SchemaConditionedSemanticOperator,
    SemanticOperatorFoundationConfig,
)
from alice_personality.n0.semantic_operator_qsre_adapter import SemanticOperatorQSREAdapter
from alice_personality.n0.full_envelope_structural_types import (
    CONTROL_RELATIONAL,
    DIRECTION_FORWARD,
    DIRECTION_REVERSE,
    FullEnvelopeOperatorState,
    ROLE_SOURCE,
    ROLE_TARGET,
    TRAVERSAL_AGGREGATE,
    TRAVERSAL_LOCAL,
    TRAVERSAL_PATH,
)


def relation_schema(count: int = 5, types: int = 4) -> DynamicRelationSchema:
    torch.manual_seed(101 + count)
    return DynamicRelationSchema(
        token_states=torch.randn(count, 3, 5, 24),
        token_mask=torch.ones(count, 5, dtype=torch.bool),
        domain_type_mask=torch.ones(count, types, dtype=torch.bool),
        range_type_mask=torch.ones(count, types, dtype=torch.bool),
        symmetric=torch.zeros(count, dtype=torch.bool),
    )


def factor_schema(count: int, seed: int) -> DynamicSemanticSchema:
    torch.manual_seed(seed)
    return DynamicSemanticSchema(
        token_states=torch.randn(count, 3, 4, 24),
        token_mask=torch.ones(count, 4, dtype=torch.bool),
    )


def operator_bundle(batch: int = 2, relations: int = 5):
    torch.manual_seed(9)
    q = torch.randn(batch, 3, 7, 24)
    qm = torch.ones(batch, 7, dtype=torch.bool)
    factors = {
        "role": factor_schema(4, 11),
        "traversal": factor_schema(3, 12),
        "direction": factor_schema(3, 13),
        "control": factor_schema(3, 14),
        "reliability": factor_schema(2, 15),
        "recency": factor_schema(2, 16),
        "temporal": factor_schema(2, 17),
        "provenance": factor_schema(2, 18),
    }
    opcodes = {
        "role": ["ROLE_SOURCE","ROLE_TARGET","ROLE_SYMMETRIC","ROLE_NONE"],
        "traversal": ["TRAVERSAL_LOCAL","TRAVERSAL_PATH","TRAVERSAL_AGGREGATE"],
        "direction": ["DIRECTION_FORWARD","DIRECTION_REVERSE","DIRECTION_BIDIRECTIONAL"],
        "control": ["CONTROL_FALLBACK","CONTROL_RELATIONAL","CONTROL_DEFER"],
        "reliability": ["MOD_RELIABILITY_OFF","MOD_RELIABILITY_ON"],
        "recency": ["MOD_RECENCY_OFF","MOD_RECENCY_ON"],
        "temporal": ["MOD_TEMPORAL_OFF","MOD_TEMPORAL_ON"],
        "provenance": ["MOD_PROVENANCE_OFF","MOD_PROVENANCE_ON"],
    }
    model = SchemaConditionedSemanticOperator(
        SemanticOperatorFoundationConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        raw = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=relation_schema(relations),
            factor_schemas=factors,
            max_steps=3,
        )
        adapted = SemanticOperatorQSREAdapter()(
            semantic_operator=raw["operator"],
            relation_schema_states=raw["relation_schema_states"],
            factor_opcodes=opcodes,
        )
    return q, qm, raw, adapted


def test_dynamic_structured_state_is_field_permutation_equivariant_and_descriptor_open() -> None:
    torch.manual_seed(21)
    model = DynamicStructuredStateV2(
        DynamicStructuredStateConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            num_layers=1,
            continuous_metadata_dim=3,
            dropout=0.0,
        )
    ).eval()
    fields = torch.randn(2, 5, 3, 4, 24)
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)
    confidence = torch.rand(2, 5)
    missing = torch.zeros(2, 5)
    metadata = torch.randn(2, 5, 3)
    type_bank = factor_schema(7, 31)
    provenance_bank = factor_schema(3, 32)
    type_idx = torch.tensor([[0,1,2,3,4],[4,3,2,1,0]])
    provenance_idx = torch.tensor([[0,1,2,0,1],[2,1,0,2,1]])
    banks = {"type": type_bank, "provenance": provenance_bank}
    indices = {"type": type_idx, "provenance": provenance_idx}

    with torch.no_grad():
        a = model(
            field_hidden_states=fields,
            field_token_mask=mask,
            field_valid_mask=valid,
            field_confidence=confidence,
            field_missing=missing,
            field_metadata=metadata,
            descriptor_banks=banks,
            descriptor_indices=indices,
        )
        perm = torch.tensor([3,0,4,1,2])
        b = model(
            field_hidden_states=fields[:, perm],
            field_token_mask=mask[:, perm],
            field_valid_mask=valid[:, perm],
            field_confidence=confidence[:, perm],
            field_missing=missing[:, perm],
            field_metadata=metadata[:, perm],
            descriptor_banks=banks,
            descriptor_indices={k:v[:, perm] for k,v in indices.items()},
        )
    assert torch.allclose(b["field_states"], a["field_states"][:, perm], atol=1e-5, rtol=1e-5)
    assert torch.allclose(b["pooled_state"], a["pooled_state"], atol=1e-5, rtol=1e-5)
    report=model.parameter_report()
    assert report["field_type_identity_parameters"]==0
    assert report["descriptor_candidate_count_dependent_parameters"]==0
    assert report["field_count_ceiling"] is None


def test_dynamic_evidence_graph_relation_permutation_is_equivariant() -> None:
    torch.manual_seed(33)
    model=DynamicSchemaEvidenceGraphV1(
        DynamicSchemaEvidenceGraphConfig(
            field_dim=24,relation_dim=24,operator_dim=24,model_dim=24,edge_metadata_dim=4,dropout=0.0
        )
    ).eval()
    batch,fields,edges,relations=2,5,6,4
    field=torch.randn(batch,fields,24)
    valid=torch.ones(batch,fields,dtype=torch.bool)
    edge_index=torch.tensor([
        [[0,1],[1,2],[2,3],[3,4],[0,3],[1,4]],
        [[0,2],[2,4],[1,3],[3,4],[0,1],[2,3]],
    ])
    edge_rel=torch.tensor([[0,1,2,3,1,0],[3,2,1,0,2,3]])
    meta=torch.randn(batch,edges,4)
    edge_valid=torch.ones(batch,edges,dtype=torch.bool)
    rel=torch.randn(batch,relations,24)
    op=torch.randn(batch,24)
    with torch.no_grad():
        a=model(
            field_state=field,field_valid_mask=valid,edge_index=edge_index,
            edge_relation_index=edge_rel,edge_metadata=meta,edge_valid_mask=edge_valid,
            relation_schema_state=rel,
            relation_mass=torch.full((batch,relations),1.0/relations),
            relation_symmetric=torch.zeros(batch,relations,dtype=torch.bool),
            semantic_activity=torch.ones(batch),
            operator_state=op,message_steps=2
        )
        perm=torch.tensor([2,0,3,1])
        inverse=torch.empty_like(perm)
        inverse[perm]=torch.arange(relations)
        uniform_mass=torch.full((batch,relations),1.0/relations)
        b=model(
            field_state=field,field_valid_mask=valid,edge_index=edge_index,
            edge_relation_index=inverse[edge_rel],edge_metadata=meta,edge_valid_mask=edge_valid,
            relation_schema_state=rel[:,perm],
            relation_mass=uniform_mass[:,perm],
            relation_symmetric=torch.zeros(batch,relations,dtype=torch.bool)[:,perm],
            semantic_activity=torch.ones(batch),
            operator_state=op,message_steps=2
        )
    assert torch.allclose(a["field_states"],b["field_states"],atol=1e-5,rtol=1e-5)
    assert torch.allclose(a["source_summary"],b["source_summary"],atol=1e-5,rtol=1e-5)
    report=model.parameter_report()
    assert report["relation_identity_parameters"]==0
    assert report["relation_count_ceiling"] is None


def test_semantic_operator_adapter_and_full_envelope_binder_executor_integrate() -> None:
    q,qm,raw,adapted=operator_bundle()
    operator=adapted["operator"]
    relation_state=adapted["relation_schema_state"]
    batch=2
    fields=5
    edges=6
    field_tokens=4
    torch.manual_seed(44)
    field_hidden=torch.randn(batch,fields,3,field_tokens,24)
    field_token_mask=torch.ones(batch,fields,field_tokens,dtype=torch.bool)

    structured=DynamicStructuredStateV2(
        DynamicStructuredStateConfig(
            semantic_dim=24,model_dim=24,num_hidden_states=3,
            num_attention_heads=4,num_layers=1,continuous_metadata_dim=3,dropout=0.0
        )
    ).eval()
    with torch.no_grad():
        structured_out=structured(
            field_hidden_states=field_hidden,
            field_token_mask=field_token_mask,
            field_valid_mask=torch.ones(batch,fields,dtype=torch.bool),
            field_confidence=torch.ones(batch,fields),
            field_missing=torch.zeros(batch,fields),
            field_metadata=torch.zeros(batch,fields,3),
            descriptor_banks={"type":factor_schema(4,55)},
            descriptor_indices={"type":torch.tensor([[0,1,2,3,0],[1,2,3,0,1]])},
        )
    field_state=structured_out["field_states"]
    field_valid=torch.ones(batch,fields,dtype=torch.bool)
    field_type=torch.tensor([[0,1,2,3,0],[1,2,3,0,1]])
    edge_index=torch.tensor([
        [[0,1],[1,2],[2,3],[3,4],[0,2],[1,4]],
        [[0,1],[1,3],[3,4],[0,2],[2,4],[1,2]],
    ])
    edge_relation=torch.tensor([[0,1,2,3,4,0],[1,2,3,4,0,1]])
    edge_valid=torch.ones(batch,edges,dtype=torch.bool)
    reliability=torch.ones(batch,edges)*0.8
    recency=torch.ones(batch,edges)*0.7
    domain=torch.ones(batch,relation_state.size(1),4,dtype=torch.bool)
    range_mask=torch.ones_like(domain)

    binder=FullEnvelopeQSREBinderV1(
        FullEnvelopeBinderConfig(
            semantic_dim=24,model_dim=24,num_hidden_states=3,edge_metadata_dim=4
        )
    ).eval()
    with torch.no_grad():
        bound=binder(
            query_hidden_states=q,
            query_token_mask=qm,
            field_hidden_states=field_hidden,
            field_token_mask=field_token_mask,
            field_state=field_state,
            field_valid_mask=field_valid,
            field_type_index=field_type,
            edge_index=edge_index,
            edge_relation_index=edge_relation,
            edge_valid_mask=edge_valid,
            edge_reliability=reliability,
            edge_recency=recency,
            relation_domain_type_mask=domain,
            relation_range_type_mask=range_mask,
            relation_symmetric=torch.zeros(batch,relation_state.size(1),dtype=torch.bool),
            relation_schema_state=relation_state,
            operator=operator,
        )

    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,model_dim=24,field_metadata_dim=3,edge_metadata_dim=4,dropout=0.0
        )
    ).eval()
    with torch.no_grad():
        executed=executor(
            field_state=field_state,
            field_metadata=torch.zeros(batch,fields,3),
            field_valid_mask=field_valid,
            edge_index=edge_index,
            edge_relation_index=edge_relation,
            edge_valid_mask=edge_valid,
            edge_support_weight=bound["edge_support_weight"],
            support_available=bound["support_available"],
            edge_reliability=reliability,
            edge_recency=recency,
            edge_temporal_match=torch.ones(batch,edges),
            edge_provenance_match=torch.ones(batch,edges),
            relation_schema_state=relation_state,
            step_relation_schema_state=adapted["step_relation_schema_state"],
            relation_symmetric=torch.zeros(batch,relation_state.size(1),dtype=torch.bool),
            operator=operator,
            focus_field_weight=bound["focus_field_weight"],
        )

    assert bound["edge_support_weight"].shape==(batch,edges)
    assert executed["relational_probability"].shape==(batch,fields)
    assert executed["relational_summary"].shape==(batch,24)
    assert binder.parameter_report()["final_layer_only_query"] is False
    assert binder.parameter_report()["exact_zero_sparse_support"] is True
    report=executor.parameter_report()
    assert report["relation_identity_parameters"]==0
    assert report["role_identity_parameters"]==0
    assert report["factor_identity_parameters"]==0


def test_dynamic_graph_and_binder_parameter_counts_do_not_depend_on_runtime_cardinality() -> None:
    graph=DynamicSchemaEvidenceGraphV1(
        DynamicSchemaEvidenceGraphConfig(field_dim=24,relation_dim=24,operator_dim=24,model_dim=24,edge_metadata_dim=4)
    )
    binder=FullEnvelopeQSREBinderV1(
        FullEnvelopeBinderConfig(semantic_dim=24,model_dim=24,num_hidden_states=3,edge_metadata_dim=4)
    )
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(field_dim=24,model_dim=24,field_metadata_dim=3,edge_metadata_dim=4)
    )
    for report in (graph.parameter_report(),binder.parameter_report(),executor.parameter_report()):
        assert report["relation_count_ceiling"] is None
    assert graph.parameter_report()["relation_count_dependent_parameters"]==0
    assert binder.parameter_report()["relation_count_dependent_parameters"]==0
    assert executor.parameter_report()["relation_count_dependent_parameters"]==0


def test_unknown_structural_opcode_fails_closed() -> None:
    q,qm,raw,_=operator_bundle(batch=1,relations=3)
    factor_opcodes = {
        "role": ["ROLE_SOURCE","ROLE_TARGET","ROLE_SYMMETRIC","ROLE_NONE"],
        "traversal": ["TRAVERSAL_LOCAL","TRAVERSAL_PATH","TRAVERSAL_AGGREGATE"],
        "direction": ["DIRECTION_FORWARD","DIRECTION_REVERSE","DIRECTION_BIDIRECTIONAL"],
        "control": ["CONTROL_FALLBACK","CONTROL_RELATIONAL","CONTROL_DEFER"],
        "reliability": ["MOD_RELIABILITY_OFF","MOD_RELIABILITY_ON"],
        "recency": ["MOD_RECENCY_OFF","MOD_RECENCY_ON"],
        "temporal": ["MOD_TEMPORAL_OFF","MOD_TEMPORAL_ON"],
        "provenance": ["MOD_PROVENANCE_OFF","MOD_PROVENANCE_ON"],
    }
    factor_opcodes["role"][0] = "ROLE_FUTURE_UNKNOWN"
    try:
        SemanticOperatorQSREAdapter()(
            semantic_operator=raw["operator"],
            relation_schema_states=raw["relation_schema_states"],
            factor_opcodes=factor_opcodes,
        )
    except ValueError as exc:
        assert "unsupported structural opcode" in str(exc)
    else:
        raise AssertionError("unknown executable primitive was silently accepted")


def _manual_operator(
    traversal_index: int,
    direction_index: int = DIRECTION_FORWARD,
    truncation_probability: float = 0.0,
) -> FullEnvelopeOperatorState:
    batch,steps,relations,dim=1,3,1,24
    relation_distribution=torch.ones(batch,steps,relations)
    if float(truncation_probability) >= 0.5:
        relation_step_mass=torch.ones(batch,steps)
        stop_probability=torch.zeros(batch,steps)
        truncation=torch.ones(batch)
    else:
        relation_step_mass=torch.tensor([[1.0,1.0,0.0]])
        stop_probability=torch.tensor([[0.0,0.0,1.0]])
        truncation=torch.zeros(batch)
    traversal=torch.zeros(batch,3)
    traversal[:,traversal_index]=1.0
    role=torch.zeros(batch,4)
    role[:,ROLE_TARGET]=1.0
    direction=torch.zeros(batch,3)
    direction[:,direction_index]=1.0
    control=torch.zeros(batch,3)
    control[:,CONTROL_RELATIONAL]=1.0
    return FullEnvelopeOperatorState(
        relation_distribution=relation_distribution,
        relation_step_mass=relation_step_mass,
        stop_probability=stop_probability,
        unknown_probability=torch.zeros(batch,steps),
        truncation_probability=truncation,
        role_distribution=role,
        traversal_distribution=traversal,
        direction_distribution=direction,
        step_direction_distribution=direction[:,None,:].expand(batch,steps,3).clone(),
        modifier_weight=torch.zeros(batch,4),
        step_modifier_weight=torch.zeros(batch,steps,4),
        applicability=torch.ones(batch),
        control_distribution=control,
        continuous_state=torch.zeros(batch,dim),
        uncertainty=torch.zeros(batch),
    )


def test_executor_local_path_aggregate_are_causally_distinct_without_hard_threshold() -> None:
    torch.manual_seed(73)
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    field_state=torch.randn(1,4,24)
    edge_index=torch.tensor([[[0,1],[1,2],[2,3]]])
    common=dict(
        field_state=field_state,
        field_metadata=torch.zeros(1,4,3),
        field_valid_mask=torch.ones(1,4,dtype=torch.bool),
        edge_index=edge_index,
        edge_relation_index=torch.zeros(1,3,dtype=torch.long),
        edge_valid_mask=torch.ones(1,3,dtype=torch.bool),
        edge_support_weight=torch.ones(1,3),
        support_available=torch.ones(1),
        edge_reliability=torch.ones(1,3),
        edge_recency=torch.ones(1,3),
        edge_temporal_match=torch.ones(1,3),
        edge_provenance_match=torch.ones(1,3),
        relation_schema_state=torch.randn(1,1,24),
        step_relation_schema_state=torch.randn(1,3,1,24),
        relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
        focus_field_weight=torch.tensor([[1.0,0.0,0.0,0.0]]),
    )
    with torch.no_grad():
        local=executor(operator=_manual_operator(TRAVERSAL_LOCAL),**common)
        path=executor(operator=_manual_operator(TRAVERSAL_PATH),**common)
        aggregate=executor(operator=_manual_operator(TRAVERSAL_AGGREGATE),**common)
    assert torch.equal(local["path_frontier"],torch.tensor([[1.0,0.0,0.0,0.0]]))
    assert float(path["path_frontier"][0,1:].sum()) > 0.0
    assert not torch.allclose(
        local["structural_role_weight"],
        aggregate["structural_role_weight"],
    )
    assert not torch.allclose(
        local["relational_summary"],
        aggregate["relational_summary"],
    )
    report=executor.parameter_report()
    assert report["continuous_traversal_mixture"] is True
    assert report["local_path_aggregate_distinct"] is True
    assert report["hard_traversal_threshold"] is False


def test_binder_can_choose_exact_no_support_even_with_type_compatible_edges() -> None:
    q,qm,raw,adapted=operator_bundle(batch=1,relations=2)
    operator=adapted["operator"]
    relation_state=adapted["relation_schema_state"]
    torch.manual_seed(97)
    binder=FullEnvelopeQSREBinderV1(
        FullEnvelopeBinderConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            edge_metadata_dim=4,
        )
    ).eval()
    with torch.no_grad():
        for parameter in binder.edge_score.parameters():
            parameter.zero_()
        binder.edge_score[-1].bias.fill_(-4.0)
        for parameter in binder.null_support_score.parameters():
            parameter.zero_()
        binder.null_support_score[-1].bias.fill_(4.0)

        out=binder(
            query_hidden_states=q,
            query_token_mask=qm,
            field_hidden_states=torch.randn(1,3,3,4,24),
            field_token_mask=torch.ones(1,3,4,dtype=torch.bool),
            field_state=torch.randn(1,3,24),
            field_valid_mask=torch.ones(1,3,dtype=torch.bool),
            field_type_index=torch.tensor([[0,1,0]]),
            edge_index=torch.tensor([[[0,1],[1,2]]]),
            edge_relation_index=torch.tensor([[0,1]]),
            edge_valid_mask=torch.ones(1,2,dtype=torch.bool),
            edge_reliability=torch.ones(1,2),
            edge_recency=torch.ones(1,2),
            relation_domain_type_mask=torch.ones(1,2,2,dtype=torch.bool),
            relation_range_type_mask=torch.ones(1,2,2,dtype=torch.bool),
            relation_symmetric=torch.zeros(1,2,dtype=torch.bool),
            relation_schema_state=relation_state,
            operator=operator,
        )
    assert torch.equal(out["edge_support_weight"], torch.zeros_like(out["edge_support_weight"]))
    assert torch.equal(out["support_available"], torch.zeros_like(out["support_available"]))
    assert torch.equal(out["null_support_mass"], torch.ones_like(out["null_support_mass"]))
    report=binder.parameter_report()
    assert report["explicit_null_support_option"] is True
    assert report["zero_support_possible_with_compatible_edges"] is True


def test_executor_zero_support_forces_zero_relational_execution_confidence() -> None:
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    common=dict(
        field_state=torch.randn(1,3,24),
        field_metadata=torch.zeros(1,3,3),
        field_valid_mask=torch.ones(1,3,dtype=torch.bool),
        edge_index=torch.tensor([[[0,1],[1,2]]]),
        edge_relation_index=torch.zeros(1,2,dtype=torch.long),
        edge_valid_mask=torch.ones(1,2,dtype=torch.bool),
        edge_support_weight=torch.zeros(1,2),
        support_available=torch.zeros(1),
        edge_reliability=torch.ones(1,2),
        edge_recency=torch.ones(1,2),
        edge_temporal_match=torch.ones(1,2),
        edge_provenance_match=torch.ones(1,2),
        relation_schema_state=torch.randn(1,1,24),
        step_relation_schema_state=torch.randn(1,3,1,24),
        relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
        focus_field_weight=torch.tensor([[1.0,0.0,0.0]]),
    )
    with torch.no_grad():
        out=executor(
            operator=_manual_operator(TRAVERSAL_PATH),
            **common,
        )
    assert torch.equal(
        out["execution_confidence"],
        torch.zeros_like(out["execution_confidence"]),
    )
    assert torch.equal(
        out["relational_probability"],
        torch.zeros_like(out["relational_probability"]),
    )
    assert torch.equal(
        out["relational_summary"],
        torch.zeros_like(out["relational_summary"]),
    )
    report=executor.parameter_report()
    assert report["execution_confidence_requires_structural_support"] is True
    assert report["zero_execution_confidence_zeroes_relational_summary"] is True


def test_binder_symmetric_relation_accepts_reversed_domain_range_types() -> None:
    domain=torch.tensor([[[True,False]]],dtype=torch.bool)
    range_mask=torch.tensor([[[False,True]]],dtype=torch.bool)
    edge_index=torch.tensor([[[0,1]]])
    edge_relation=torch.tensor([[0]])
    field_types=torch.tensor([[1,0]])
    edge_valid=torch.ones(1,1,dtype=torch.bool)

    directed=FullEnvelopeQSREBinderV1._type_compatibility(
        relation_domain_type_mask=domain,
        relation_range_type_mask=range_mask,
        relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
        edge_relation_index=edge_relation,
        edge_index=edge_index,
        field_type_index=field_types,
        edge_valid_mask=edge_valid,
    )
    symmetric=FullEnvelopeQSREBinderV1._type_compatibility(
        relation_domain_type_mask=domain,
        relation_range_type_mask=range_mask,
        relation_symmetric=torch.ones(1,1,dtype=torch.bool),
        edge_relation_index=edge_relation,
        edge_index=edge_index,
        field_type_index=field_types,
        edge_valid_mask=edge_valid,
    )
    assert directed.item() is False
    assert symmetric.item() is True


def test_dynamic_graph_symmetric_edge_reversal_is_invariant() -> None:
    torch.manual_seed(811)
    graph=DynamicSchemaEvidenceGraphV1(
        DynamicSchemaEvidenceGraphConfig(
            field_dim=24,
            relation_dim=24,
            operator_dim=24,
            model_dim=24,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    field=torch.randn(1,4,24)
    valid=torch.ones(1,4,dtype=torch.bool)
    edge=torch.tensor([[[0,1],[1,2],[2,3]]])
    reversed_edge=edge.flip(-1)
    relation_index=torch.zeros(1,3,dtype=torch.long)
    metadata=torch.zeros(1,3,4)
    edge_valid=torch.ones(1,3,dtype=torch.bool)
    relation_state=torch.randn(1,1,24)
    relation_mass=torch.ones(1,1)
    symmetric=torch.ones(1,1,dtype=torch.bool)
    operator_state=torch.randn(1,24)
    with torch.no_grad():
        a=graph(
            field_state=field,
            field_valid_mask=valid,
            edge_index=edge,
            edge_relation_index=relation_index,
            edge_metadata=metadata,
            edge_valid_mask=edge_valid,
            relation_schema_state=relation_state,
            relation_mass=relation_mass,
            relation_symmetric=symmetric,
            semantic_activity=torch.ones(1),
            operator_state=operator_state,
            message_steps=2,
        )
        b=graph(
            field_state=field,
            field_valid_mask=valid,
            edge_index=reversed_edge,
            edge_relation_index=relation_index,
            edge_metadata=metadata,
            edge_valid_mask=edge_valid,
            relation_schema_state=relation_state,
            relation_mass=relation_mass,
            relation_symmetric=symmetric,
            semantic_activity=torch.ones(1),
            operator_state=operator_state,
            message_steps=2,
        )
    assert torch.allclose(a["field_states"],b["field_states"],atol=1e-5,rtol=1e-5)
    assert torch.allclose(a["source_summary"],b["source_summary"],atol=1e-5,rtol=1e-5)
    assert torch.allclose(a["target_summary"],b["target_summary"],atol=1e-5,rtol=1e-5)
    report=graph.parameter_report()
    assert report["symmetric_edge_endpoint_order_invariant"] is True


def test_executor_symmetric_relation_ignores_forward_reverse_storage_orientation() -> None:
    torch.manual_seed(812)
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    common=dict(
        field_state=torch.randn(1,3,24),
        field_metadata=torch.zeros(1,3,3),
        field_valid_mask=torch.ones(1,3,dtype=torch.bool),
        edge_index=torch.tensor([[[0,1],[1,2]]]),
        edge_relation_index=torch.zeros(1,2,dtype=torch.long),
        edge_valid_mask=torch.ones(1,2,dtype=torch.bool),
        edge_support_weight=torch.ones(1,2),
        support_available=torch.ones(1),
        edge_reliability=torch.ones(1,2),
        edge_recency=torch.ones(1,2),
        edge_temporal_match=torch.ones(1,2),
        edge_provenance_match=torch.ones(1,2),
        relation_schema_state=torch.randn(1,1,24),
        step_relation_schema_state=torch.randn(1,3,1,24),
        relation_symmetric=torch.ones(1,1,dtype=torch.bool),
        focus_field_weight=torch.tensor([[1.0,0.0,0.0]]),
    )
    with torch.no_grad():
        forward=executor(
            operator=_manual_operator(TRAVERSAL_PATH,DIRECTION_FORWARD),
            **common,
        )
        reverse=executor(
            operator=_manual_operator(TRAVERSAL_PATH,DIRECTION_REVERSE),
            **common,
        )
        reversed_storage=dict(common)
        reversed_storage["edge_index"]=common["edge_index"].flip(-1)
        storage_flip=executor(
            operator=_manual_operator(TRAVERSAL_PATH,DIRECTION_FORWARD),
            **reversed_storage,
        )
    assert torch.allclose(
        forward["path_frontier"],
        reverse["path_frontier"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        forward["relational_summary"],
        reverse["relational_summary"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        forward["node_state"],
        storage_flip["node_state"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        forward["relational_summary"],
        storage_flip["relational_summary"],
        atol=1e-5,
        rtol=1e-5,
    )
    report=executor.parameter_report()
    assert report["symmetric_relation_direction_collapses_to_bidirectional"] is True
    assert report["symmetric_relation_endpoint_order_invariant"] is True


def test_executor_truncated_program_cannot_claim_relational_execution_confidence() -> None:
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    common=dict(
        field_state=torch.randn(1,3,24),
        field_metadata=torch.zeros(1,3,3),
        field_valid_mask=torch.ones(1,3,dtype=torch.bool),
        edge_index=torch.tensor([[[0,1],[1,2]]]),
        edge_relation_index=torch.zeros(1,2,dtype=torch.long),
        edge_valid_mask=torch.ones(1,2,dtype=torch.bool),
        edge_support_weight=torch.ones(1,2),
        support_available=torch.ones(1),
        edge_reliability=torch.ones(1,2),
        edge_recency=torch.ones(1,2),
        edge_temporal_match=torch.ones(1,2),
        edge_provenance_match=torch.ones(1,2),
        relation_schema_state=torch.randn(1,1,24),
        step_relation_schema_state=torch.randn(1,3,1,24),
        relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
        focus_field_weight=torch.tensor([[1.0,0.0,0.0]]),
    )
    with torch.no_grad():
        complete=executor(
            operator=_manual_operator(
                TRAVERSAL_PATH,
                DIRECTION_FORWARD,
                truncation_probability=0.0,
            ),
            **common,
        )
        truncated=executor(
            operator=_manual_operator(
                TRAVERSAL_PATH,
                DIRECTION_FORWARD,
                truncation_probability=1.0,
            ),
            **common,
        )
    assert float(complete["execution_confidence"].item()) > 0.0
    assert torch.equal(
        truncated["execution_confidence"],
        torch.zeros_like(truncated["execution_confidence"]),
    )
    assert torch.equal(
        truncated["relational_probability"],
        torch.zeros_like(truncated["relational_probability"]),
    )
    assert executor.parameter_report()["execution_confidence_requires_program_completion"] is True


def test_binder_allows_minus_one_type_for_padded_fields_but_not_valid_edge_endpoints() -> None:
    domain=torch.ones(1,1,2,dtype=torch.bool)
    range_mask=torch.ones_like(domain)
    relation_symmetric=torch.zeros(1,1,dtype=torch.bool)
    field_types=torch.tensor([[0,1,-1]])
    edge_relation=torch.zeros(1,1,dtype=torch.long)

    compatible=FullEnvelopeQSREBinderV1._type_compatibility(
        relation_domain_type_mask=domain,
        relation_range_type_mask=range_mask,
        relation_symmetric=relation_symmetric,
        edge_relation_index=edge_relation,
        edge_index=torch.tensor([[[0,1]]]),
        field_type_index=field_types,
        edge_valid_mask=torch.ones(1,1,dtype=torch.bool),
    )
    assert compatible.item() is True

    try:
        FullEnvelopeQSREBinderV1._type_compatibility(
            relation_domain_type_mask=domain,
            relation_range_type_mask=range_mask,
            relation_symmetric=relation_symmetric,
            edge_relation_index=edge_relation,
            edge_index=torch.tensor([[[1,2]]]),
            field_type_index=field_types,
            edge_valid_mask=torch.ones(1,1,dtype=torch.bool),
        )
    except ValueError as exc:
        assert "without a runtime type" in str(exc)
    else:
        raise AssertionError("valid edge to padded untyped field did not fail closed")


def test_executor_mixed_forward_then_reverse_tracks_final_semantic_source() -> None:
    torch.manual_seed(913)
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    relation_distribution=torch.tensor(
        [[[1.0,0.0],[0.0,1.0],[0.5,0.5]]]
    )
    step_direction=torch.zeros(1,3,3)
    step_direction[0,0,DIRECTION_FORWARD]=1.0
    step_direction[0,1,DIRECTION_REVERSE]=1.0
    step_direction[0,2,DIRECTION_FORWARD]=1.0
    role=torch.zeros(1,4)
    role[0,ROLE_SOURCE]=1.0
    traversal=torch.zeros(1,3)
    traversal[0,TRAVERSAL_PATH]=1.0
    control=torch.zeros(1,3)
    control[0,CONTROL_RELATIONAL]=1.0
    operator=FullEnvelopeOperatorState(
        relation_distribution=relation_distribution,
        relation_step_mass=torch.tensor([[1.0,1.0,0.0]]),
        stop_probability=torch.tensor([[0.0,0.0,1.0]]),
        unknown_probability=torch.zeros(1,3),
        truncation_probability=torch.zeros(1),
        role_distribution=role,
        traversal_distribution=traversal,
        direction_distribution=torch.tensor([[0.5,0.5,0.0]]),
        step_direction_distribution=step_direction,
        modifier_weight=torch.zeros(1,4),
        step_modifier_weight=torch.zeros(1,3,4),
        applicability=torch.ones(1),
        control_distribution=control,
        continuous_state=torch.zeros(1,24),
        uncertainty=torch.zeros(1),
    )
    with torch.no_grad():
        out=executor(
            field_state=torch.randn(1,3,24),
            field_metadata=torch.zeros(1,3,3),
            field_valid_mask=torch.ones(1,3,dtype=torch.bool),
            edge_index=torch.tensor([[[0,1],[2,1]]]),
            edge_relation_index=torch.tensor([[0,1]]),
            edge_valid_mask=torch.ones(1,2,dtype=torch.bool),
            edge_support_weight=torch.ones(1,2),
            support_available=torch.ones(1),
            edge_reliability=torch.ones(1,2),
            edge_recency=torch.ones(1,2),
            edge_temporal_match=torch.ones(1,2),
            edge_provenance_match=torch.ones(1,2),
            relation_schema_state=torch.randn(1,2,24),
            step_relation_schema_state=torch.randn(1,3,2,24),
            relation_symmetric=torch.zeros(1,2,dtype=torch.bool),
            operator=operator,
            focus_field_weight=torch.tensor([[1.0,0.0,0.0]]),
        )
    assert int(out["relational_probability"].argmax(dim=-1).item()) == 2
    assert int(out["path_semantic_source_weight"].argmax(dim=-1).item()) == 2
    assert executor.parameter_report()["semantic_endpoint_tracking_is_step_conditioned"] is True
    assert executor.parameter_report()["global_direction_not_used_for_final_path_role"] is True


def test_adapter_allows_semantic_only_factor_banks_without_structural_opcode_mapping() -> None:
    torch.manual_seed(131)
    q = torch.randn(1,3,7,24)
    qm = torch.ones(1,7,dtype=torch.bool)
    factors = {
        "role": factor_schema(4,132),
        "traversal": factor_schema(3,133),
        "direction": factor_schema(3,134),
        "control": factor_schema(3,135),
        "reliability": factor_schema(2,136),
        "recency": factor_schema(2,137),
        "temporal": factor_schema(2,138),
        "provenance": factor_schema(2,139),
        "open_semantic_factor": factor_schema(5,140),
    }
    model = SchemaConditionedSemanticOperator(
        SemanticOperatorFoundationConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        raw = model(
            query_hidden_states=q,
            query_token_mask=qm,
            relation_schema=relation_schema(4),
            factor_schemas=factors,
            max_steps=3,
        )
        adapted = SemanticOperatorQSREAdapter()(
            semantic_operator=raw["operator"],
            relation_schema_states=raw["relation_schema_states"],
            factor_opcodes={
                "role": ["ROLE_SOURCE","ROLE_TARGET","ROLE_SYMMETRIC","ROLE_NONE"],
                "traversal": ["TRAVERSAL_LOCAL","TRAVERSAL_PATH","TRAVERSAL_AGGREGATE"],
                "direction": ["DIRECTION_FORWARD","DIRECTION_REVERSE","DIRECTION_BIDIRECTIONAL"],
                "control": ["CONTROL_FALLBACK","CONTROL_RELATIONAL","CONTROL_DEFER"],
                "reliability": ["MOD_RELIABILITY_OFF","MOD_RELIABILITY_ON"],
                "recency": ["MOD_RECENCY_OFF","MOD_RECENCY_ON"],
                "temporal": ["MOD_TEMPORAL_OFF","MOD_TEMPORAL_ON"],
                "provenance": ["MOD_PROVENANCE_OFF","MOD_PROVENANCE_ON"],
            },
        )
    assert "open_semantic_factor" in raw["operator"].factor_distributions
    assert adapted["operator"].continuous_state.shape == (1,24)
    report = SemanticOperatorQSREAdapter().parameter_report()
    assert report["semantic_only_factor_banks_supported"] is True
    assert report["structural_factor_mapping_may_be_subset_of_semantic_banks"] is True
    assert report["runtime_semantic_factor_bank_ceiling"] is None


def test_structured_descriptor_bank_count_does_not_rescale_duplicate_semantics() -> None:
    torch.manual_seed(151)
    model = DynamicStructuredStateV2(
        DynamicStructuredStateConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            num_layers=1,
            continuous_metadata_dim=3,
            dropout=0.0,
        )
    ).eval()
    fields = torch.randn(1,4,3,5,24)
    token_mask = torch.ones(1,4,5,dtype=torch.bool)
    valid = torch.ones(1,4,dtype=torch.bool)
    confidence = torch.ones(1,4)
    missing = torch.zeros(1,4)
    metadata = torch.zeros(1,4,3)
    bank = factor_schema(4,152)
    index = torch.tensor([[0,1,2,3]])
    with torch.no_grad():
        one = model(
            field_hidden_states=fields,
            field_token_mask=token_mask,
            field_valid_mask=valid,
            field_confidence=confidence,
            field_missing=missing,
            field_metadata=metadata,
            descriptor_banks={"a":bank},
            descriptor_indices={"a":index},
        )
        duplicate = model(
            field_hidden_states=fields,
            field_token_mask=token_mask,
            field_valid_mask=valid,
            field_confidence=confidence,
            field_missing=missing,
            field_metadata=metadata,
            descriptor_banks={"a":bank,"b":bank},
            descriptor_indices={"a":index,"b":index},
        )
    assert torch.allclose(
        one["field_states"],
        duplicate["field_states"],
        atol=1e-6,
        rtol=1e-6,
    )
    report=model.parameter_report()
    assert report["descriptor_bank_count_normalized"] is True
    assert report["global_static_layer_mixture"] is False


def test_reopened_structured_evidence_and_binder_have_no_global_static_layer_logits() -> None:
    structured = DynamicStructuredStateV2(
        DynamicStructuredStateConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            num_layers=1,
            continuous_metadata_dim=3,
            dropout=0.0,
        )
    )
    from alice_personality.n0.dynamic_evidence_view_v2 import (
        DynamicEvidenceViewConfig,
        DynamicEvidenceViewV2,
    )
    evidence = DynamicEvidenceViewV2(
        DynamicEvidenceViewConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            dropout=0.0,
        )
    )
    binder = FullEnvelopeQSREBinderV1(
        FullEnvelopeBinderConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            edge_metadata_dim=4,
        )
    )
    for module in (structured,evidence,binder):
        names={name for name,_ in module.named_parameters()}
        assert "layer_logits" not in names
        report=module.parameter_report()
        assert report["global_static_layer_mixture"] is False


def test_dynamic_graph_zero_semantic_activity_cannot_inject_relation_messages() -> None:
    torch.manual_seed(171)
    graph = DynamicSchemaEvidenceGraphV1(
        DynamicSchemaEvidenceGraphConfig(
            field_dim=24,
            relation_dim=24,
            operator_dim=24,
            model_dim=24,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    common = dict(
        field_state=torch.randn(1,4,24),
        field_valid_mask=torch.ones(1,4,dtype=torch.bool),
        edge_index=torch.tensor([[[0,1],[1,2],[2,3]]]),
        edge_relation_index=torch.zeros(1,3,dtype=torch.long),
        edge_metadata=torch.randn(1,3,4),
        edge_valid_mask=torch.ones(1,3,dtype=torch.bool),
        relation_schema_state=torch.randn(1,1,24),
        relation_mass=torch.ones(1,1),
        relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
        semantic_activity=torch.zeros(1),
        operator_state=torch.randn(1,24),
    )
    with torch.no_grad():
        one = graph(message_steps=1, **common)
        four = graph(message_steps=4, **common)
    assert torch.allclose(
        one["field_states"],
        four["field_states"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        one["source_summary"],
        torch.zeros_like(one["source_summary"]),
    )
    assert torch.equal(
        one["target_summary"],
        torch.zeros_like(one["target_summary"]),
    )
    report = graph.parameter_report()
    assert report["soft_relational_activity_gate"] is True
    assert report["zero_activity_preserves_pre_message_graph_state"] is True


def test_empty_evidence_graph_produces_zero_graph_summaries() -> None:
    torch.manual_seed(201)
    graph=DynamicSchemaEvidenceGraphV1(
        DynamicSchemaEvidenceGraphConfig(
            field_dim=24,
            relation_dim=24,
            operator_dim=24,
            model_dim=24,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        out=graph(
            field_state=torch.randn(1,3,24),
            field_valid_mask=torch.ones(1,3,dtype=torch.bool),
            edge_index=torch.empty(1,0,2,dtype=torch.long),
            edge_relation_index=torch.empty(1,0,dtype=torch.long),
            edge_metadata=torch.empty(1,0,4),
            edge_valid_mask=torch.empty(1,0,dtype=torch.bool),
            relation_schema_state=torch.randn(1,2,24),
            relation_mass=torch.tensor([[0.6,0.4]]),
            relation_symmetric=torch.zeros(1,2,dtype=torch.bool),
            semantic_activity=torch.ones(1),
            operator_state=torch.randn(1,24),
            message_steps=3,
        )
    assert torch.equal(
        out["source_summary"],
        torch.zeros_like(out["source_summary"]),
    )
    assert torch.equal(
        out["target_summary"],
        torch.zeros_like(out["target_summary"]),
    )
    assert torch.equal(
        out["graph_support_activity"],
        torch.zeros_like(out["graph_support_activity"]),
    )
    assert graph.parameter_report()["zero_edge_graph_summary_is_zero"] is True


def test_executor_confidence_uses_program_start_probability_not_expected_step_count() -> None:
    torch.manual_seed(221)
    batch,steps,relations,dim=1,3,1,24
    role=torch.zeros(batch,4)
    role[:,ROLE_TARGET]=1.0
    traversal=torch.zeros(batch,3)
    traversal[:,TRAVERSAL_LOCAL]=1.0
    direction=torch.zeros(batch,steps,3)
    direction[:,:,DIRECTION_FORWARD]=1.0
    control=torch.zeros(batch,3)
    control[:,CONTROL_RELATIONAL]=1.0
    operator=FullEnvelopeOperatorState(
        relation_distribution=torch.ones(batch,steps,relations),
        relation_step_mass=torch.tensor([[0.2,0.2,0.0]]),
        stop_probability=torch.tensor([[0.8,0.0,0.2]]),
        unknown_probability=torch.zeros(batch,steps),
        truncation_probability=torch.zeros(batch),
        role_distribution=role,
        traversal_distribution=traversal,
        direction_distribution=direction[:,0],
        step_direction_distribution=direction,
        modifier_weight=torch.zeros(batch,4),
        step_modifier_weight=torch.zeros(batch,steps,4),
        applicability=torch.ones(batch),
        control_distribution=control,
        continuous_state=torch.zeros(batch,dim),
        uncertainty=torch.zeros(batch),
    )
    operator.validate(relation_count=relations,model_dim=dim)
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        out=executor(
            field_state=torch.randn(1,2,24),
            field_metadata=torch.zeros(1,2,3),
            field_valid_mask=torch.ones(1,2,dtype=torch.bool),
            edge_index=torch.tensor([[[0,1]]]),
            edge_relation_index=torch.zeros(1,1,dtype=torch.long),
            edge_valid_mask=torch.ones(1,1,dtype=torch.bool),
            edge_support_weight=torch.ones(1,1),
            support_available=torch.ones(1),
            edge_reliability=torch.ones(1,1),
            edge_recency=torch.ones(1,1),
            edge_temporal_match=torch.ones(1,1),
            edge_provenance_match=torch.ones(1,1),
            relation_schema_state=torch.randn(1,1,24),
            step_relation_schema_state=torch.randn(1,3,1,24),
            relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
            operator=operator,
            focus_field_weight=torch.tensor([[1.0,0.0]]),
        )
    assert torch.allclose(
        out["execution_confidence"],
        torch.tensor([0.2]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert executor.parameter_report()[
        "execution_confidence_uses_program_start_probability_not_expected_step_count"
    ] is True


def test_structured_state_rejects_out_of_range_confidence_metadata() -> None:
    model=DynamicStructuredStateV2(
        DynamicStructuredStateConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            num_layers=1,
            continuous_metadata_dim=3,
            dropout=0.0,
        )
    )
    try:
        model(
            field_hidden_states=torch.randn(1,2,3,4,24),
            field_token_mask=torch.ones(1,2,4,dtype=torch.bool),
            field_valid_mask=torch.ones(1,2,dtype=torch.bool),
            field_confidence=torch.tensor([[1.1,0.5]]),
            field_missing=torch.zeros(1,2),
            field_metadata=torch.zeros(1,2,3),
            descriptor_banks={},
            descriptor_indices={},
        )
    except ValueError as exc:
        assert "field_confidence" in str(exc)
        assert "[0,1]" in str(exc)
    else:
        raise AssertionError("out-of-range structured confidence did not fail closed")


def test_executor_rejects_out_of_range_support_metadata() -> None:
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    )
    try:
        executor(
            field_state=torch.randn(1,2,24),
            field_metadata=torch.zeros(1,2,3),
            field_valid_mask=torch.ones(1,2,dtype=torch.bool),
            edge_index=torch.tensor([[[0,1]]]),
            edge_relation_index=torch.zeros(1,1,dtype=torch.long),
            edge_valid_mask=torch.ones(1,1,dtype=torch.bool),
            edge_support_weight=torch.tensor([[1.2]]),
            support_available=torch.ones(1),
            edge_reliability=torch.ones(1,1),
            edge_recency=torch.ones(1,1),
            edge_temporal_match=torch.ones(1,1),
            edge_provenance_match=torch.ones(1,1),
            relation_schema_state=torch.randn(1,1,24),
            step_relation_schema_state=torch.randn(1,3,1,24),
            relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
            operator=_manual_operator(TRAVERSAL_LOCAL),
            focus_field_weight=torch.tensor([[1.0,0.0]]),
        )
    except ValueError as exc:
        assert "edge_support_weight" in str(exc)
        assert "[0,1]" in str(exc)
    else:
        raise AssertionError("out-of-range support weight did not fail closed")


def test_valid_edge_cannot_reference_padded_invalid_field() -> None:
    graph=DynamicSchemaEvidenceGraphV1(
        DynamicSchemaEvidenceGraphConfig(
            field_dim=24,
            relation_dim=24,
            operator_dim=24,
            model_dim=24,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    )
    try:
        graph(
            field_state=torch.randn(1,3,24),
            field_valid_mask=torch.tensor([[True,True,False]]),
            edge_index=torch.tensor([[[1,2]]]),
            edge_relation_index=torch.zeros(1,1,dtype=torch.long),
            edge_metadata=torch.zeros(1,1,4),
            edge_valid_mask=torch.ones(1,1,dtype=torch.bool),
            relation_schema_state=torch.randn(1,1,24),
            relation_mass=torch.ones(1,1),
            relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
            semantic_activity=torch.ones(1),
            operator_state=torch.randn(1,24),
            message_steps=1,
        )
    except ValueError as exc:
        assert "padded invalid field" in str(exc)
    else:
        raise AssertionError("valid edge into padded field did not fail closed")


def test_structured_padded_fields_are_inert_even_with_arbitrary_hidden_content() -> None:
    torch.manual_seed(241)
    model=DynamicStructuredStateV2(
        DynamicStructuredStateConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            num_layers=1,
            continuous_metadata_dim=3,
            dropout=0.0,
        )
    ).eval()
    hidden=torch.randn(1,5,3,4,24)
    changed=hidden.clone()
    changed[:,3:] = torch.randn_like(changed[:,3:]) * 100.0
    token_mask=torch.ones(1,5,4,dtype=torch.bool)
    token_mask[:,3:]=False
    valid=torch.tensor([[True,True,True,False,False]])
    bank=factor_schema(3,242)
    indices=torch.tensor([[0,1,2,-1,-1]])
    common=dict(
        field_token_mask=token_mask,
        field_valid_mask=valid,
        field_confidence=torch.tensor([[0.9,0.8,0.7,0.0,0.0]]),
        field_missing=torch.tensor([[0.0,0.0,0.0,1.0,1.0]]),
        field_metadata=torch.zeros(1,5,3),
        descriptor_banks={"type":bank},
        descriptor_indices={"type":indices},
    )
    with torch.no_grad():
        a=model(field_hidden_states=hidden,**common)
        b=model(field_hidden_states=changed,**common)
    assert torch.allclose(
        a["field_states"][:,:3],
        b["field_states"][:,:3],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        a["field_states"][:,3:],
        torch.zeros_like(a["field_states"][:,3:]),
    )
    assert torch.allclose(
        a["pooled_state"],
        b["pooled_state"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_executor_uses_step_conditioned_relation_schema_state_for_ordered_execution() -> None:
    torch.manual_seed(261)
    executor=FullEnvelopeQSREExecutorV1(
        FullEnvelopeExecutorConfig(
            field_dim=24,
            model_dim=24,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    operator=_manual_operator(TRAVERSAL_PATH,DIRECTION_FORWARD)
    global_relation=torch.randn(1,1,24)
    step_a=global_relation[:,None,:,:].expand(1,3,1,24).clone()
    step_b=step_a.clone()
    step_b[:,1] = step_b[:,1] + 3.0 * torch.randn_like(step_b[:,1])
    common=dict(
        field_state=torch.randn(1,3,24),
        field_metadata=torch.zeros(1,3,3),
        field_valid_mask=torch.ones(1,3,dtype=torch.bool),
        edge_index=torch.tensor([[[0,1],[1,2]]]),
        edge_relation_index=torch.zeros(1,2,dtype=torch.long),
        edge_valid_mask=torch.ones(1,2,dtype=torch.bool),
        edge_support_weight=torch.ones(1,2),
        support_available=torch.ones(1),
        edge_reliability=torch.ones(1,2),
        edge_recency=torch.ones(1,2),
        edge_temporal_match=torch.ones(1,2),
        edge_provenance_match=torch.ones(1,2),
        relation_schema_state=global_relation,
        relation_symmetric=torch.zeros(1,1,dtype=torch.bool),
        operator=operator,
        focus_field_weight=torch.tensor([[1.0,0.0,0.0]]),
    )
    with torch.no_grad():
        a=executor(step_relation_schema_state=step_a,**common)
        b=executor(step_relation_schema_state=step_b,**common)
    assert not torch.allclose(a["node_state"],b["node_state"])
    assert not torch.allclose(a["relational_summary"],b["relational_summary"])
    report=executor.parameter_report()
    assert report["step_conditioned_relation_schema_state"] is True
    assert report["global_relation_state_not_used_for_step_edge_features"] is True


def test_dynamic_graph_requires_relation_mass_supported_by_present_edges() -> None:
    torch.manual_seed(271)
    graph=DynamicSchemaEvidenceGraphV1(
        DynamicSchemaEvidenceGraphConfig(
            field_dim=24,
            relation_dim=24,
            operator_dim=24,
            model_dim=24,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    common=dict(
        field_state=torch.randn(1,3,24),
        field_valid_mask=torch.ones(1,3,dtype=torch.bool),
        edge_index=torch.tensor([[[0,1],[1,2]]]),
        edge_relation_index=torch.zeros(1,2,dtype=torch.long),
        edge_metadata=torch.zeros(1,2,4),
        edge_valid_mask=torch.ones(1,2,dtype=torch.bool),
        relation_schema_state=torch.randn(1,2,24),
        relation_symmetric=torch.zeros(1,2,dtype=torch.bool),
        semantic_activity=torch.ones(1),
        operator_state=torch.randn(1,24),
        message_steps=2,
    )
    with torch.no_grad():
        unsupported=graph(
            relation_mass=torch.tensor([[0.0,1.0]]),
            **common,
        )
        supported=graph(
            relation_mass=torch.tensor([[0.65,0.35]]),
            **common,
        )
    assert torch.equal(
        unsupported["graph_support_activity"],
        torch.zeros_like(unsupported["graph_support_activity"]),
    )
    assert torch.equal(
        unsupported["source_summary"],
        torch.zeros_like(unsupported["source_summary"]),
    )
    assert torch.equal(
        unsupported["target_summary"],
        torch.zeros_like(unsupported["target_summary"]),
    )
    assert torch.allclose(
        supported["supported_relation_mass"],
        torch.tensor([0.65]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        supported["graph_support_activity"],
        torch.tensor([0.65]),
        atol=1e-6,
        rtol=1e-6,
    )
    report=graph.parameter_report()
    assert report["unsupported_relation_mass_cannot_activate_graph_views"] is True
    assert report["duplicate_edges_do_not_inflate_supported_relation_mass"] is True
