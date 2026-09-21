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
            operator_state=op,message_steps=2
        )
        perm=torch.tensor([2,0,3,1])
        inverse=torch.empty_like(perm)
        inverse[perm]=torch.arange(relations)
        b=model(
            field_state=field,field_valid_mask=valid,edge_index=edge_index,
            edge_relation_index=inverse[edge_rel],edge_metadata=meta,edge_valid_mask=edge_valid,
            relation_schema_state=rel[:,perm],operator_state=op,message_steps=2
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
            edge_reliability=reliability,
            edge_recency=recency,
            edge_temporal_match=torch.ones(batch,edges),
            edge_provenance_match=torch.ones(batch,edges),
            relation_schema_state=relation_state,
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
