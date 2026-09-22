from __future__ import annotations

from dataclasses import replace

import torch

from alice_personality.n0.full_envelope_structural_types import (
    CONTROL_RELATIONAL,
    DIRECTION_FORWARD,
    MOD_PROVENANCE_CONSTRAINT,
    MOD_RECENCY,
    MOD_RELIABILITY,
    MOD_TEMPORAL_CONSTRAINT,
    ROLE_TARGET,
    TRAVERSAL_LOCAL,
    TRAVERSAL_PATH,
    FullEnvelopeOperatorState,
)
from alice_personality.n0.qsre_full_envelope_binder_v1 import (
    FullEnvelopeBinderConfig,
    FullEnvelopeQSREBinderV1,
)
from alice_personality.n0.qsre_full_envelope_executor_v1 import (
    FullEnvelopeExecutorConfig,
    FullEnvelopeQSREExecutorV1,
)
from alice_personality.n0.n0_full_envelope_stack_v1 import (
    N0FullEnvelopeStackConfig,
    N0FullEnvelopeStackV1,
)
from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
)
from alice_personality.n0.semantic_operator_qsre_adapter import (
    SemanticOperatorQSREAdapter,
)


DIM = 24


def _operator(*, modifier_index: int | None = None) -> FullEnvelopeOperatorState:
    modifier = torch.zeros(1, 4)
    step_modifier = torch.zeros(1, 2, 4)
    if modifier_index is not None:
        modifier[:, modifier_index] = 1.0
        step_modifier[:, :, modifier_index] = 1.0

    role = torch.zeros(1, 4)
    role[:, ROLE_TARGET] = 1.0
    traversal = torch.zeros(1, 3)
    traversal[:, TRAVERSAL_LOCAL] = 1.0
    direction = torch.zeros(1, 3)
    direction[:, DIRECTION_FORWARD] = 1.0
    control = torch.zeros(1, 3)
    control[:, CONTROL_RELATIONAL] = 1.0

    return FullEnvelopeOperatorState(
        relation_distribution=torch.ones(1, 2, 1),
        relation_step_mass=torch.tensor([[1.0, 0.0]]),
        stop_probability=torch.tensor([[0.0, 1.0]]),
        unknown_probability=torch.zeros(1, 2),
        truncation_probability=torch.zeros(1),
        role_distribution=role,
        traversal_distribution=traversal,
        direction_distribution=direction,
        step_direction_distribution=direction[:, None, :].expand(1, 2, 3).clone(),
        modifier_weight=modifier,
        step_modifier_weight=step_modifier,
        applicability=torch.ones(1),
        control_distribution=control,
        continuous_state=torch.zeros(1, DIM),
        uncertainty=torch.zeros(1),
    )


class _FeatureProbe(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self.index : self.index + 1]


class _FeatureStateProbe(torch.nn.Module):
    def __init__(self, index: int, width: int) -> None:
        super().__init__()
        self.index = int(index)
        self.width = int(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        selected = value[..., self.index : self.index + 1]
        return selected.expand(*selected.shape[:-1], self.width)


def _binder_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(601)
    return {
        "query_hidden_states": torch.randn(1, 3, 5, DIM),
        "query_token_mask": torch.ones(1, 5, dtype=torch.bool),
        "field_hidden_states": torch.randn(1, 2, 3, 4, DIM),
        "field_token_mask": torch.ones(1, 2, 4, dtype=torch.bool),
        "field_state": torch.randn(1, 2, DIM),
        "field_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "field_type_index": torch.zeros(1, 2, dtype=torch.long),
        "edge_index": torch.tensor([[[0, 1], [0, 1]]]),
        "edge_relation_index": torch.zeros(1, 2, dtype=torch.long),
        "edge_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "relation_domain_type_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "relation_range_type_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "relation_symmetric": torch.zeros(1, 1, dtype=torch.bool),
        "relation_schema_state": torch.randn(1, 1, DIM),
    }


def test_binder_support_remains_neutral_to_reliability_and_recency_criteria() -> None:
    """Binder owns candidate support; step-local criterion arbitration is downstream."""
    base = _binder_inputs()
    off = _operator()

    for scalar_offset, modifier_index in (
        (1, MOD_RELIABILITY),
        (2, MOD_RECENCY),
    ):
        binder = FullEnvelopeQSREBinderV1(
            FullEnvelopeBinderConfig(
                semantic_dim=DIM,
                model_dim=DIM,
                num_hidden_states=3,
                edge_metadata_dim=4,
            )
        ).eval()
        # Binder edge-score feature = five D-wide semantic blocks followed by
        # [relation_mass, reliability, recency, applicability, ...].
        binder.edge_score = _FeatureProbe(5 * DIM + scalar_offset)

        kwargs = dict(base)
        kwargs["edge_reliability"] = torch.tensor([[0.1, 0.9]])
        kwargs["edge_recency"] = torch.tensor([[0.2, 0.8]])

        with torch.no_grad():
            disabled = binder(operator=off, **kwargs)
            enabled = binder(
                operator=_operator(modifier_index=modifier_index),
                **kwargs,
            )

        assert torch.allclose(
            disabled["support_logits"][:, 0],
            disabled["support_logits"][:, 1],
            atol=1.0e-7,
            rtol=0.0,
        )
        assert torch.allclose(
            enabled["support_logits"][:, 0],
            enabled["support_logits"][:, 1],
            atol=1.0e-7,
            rtol=0.0,
        )


def _mixed_step_same_relation_operator() -> FullEnvelopeOperatorState:
    """One relation is reused across two steps with reliability ON then OFF."""
    modifier=torch.zeros(1,4)
    modifier[:,MOD_RELIABILITY]=1.0
    step_modifier=torch.zeros(1,3,4)
    step_modifier[:,0,MOD_RELIABILITY]=1.0

    role=torch.zeros(1,4)
    role[:,ROLE_TARGET]=1.0
    traversal=torch.zeros(1,3)
    traversal[:,TRAVERSAL_PATH]=1.0
    direction=torch.zeros(1,3)
    direction[:,DIRECTION_FORWARD]=1.0
    control=torch.zeros(1,3)
    control[:,CONTROL_RELATIONAL]=1.0

    return FullEnvelopeOperatorState(
        relation_distribution=torch.ones(1,3,1),
        relation_step_mass=torch.tensor([[1.0,1.0,0.0]]),
        stop_probability=torch.tensor([[0.0,0.0,1.0]]),
        unknown_probability=torch.zeros(1,3),
        truncation_probability=torch.zeros(1),
        role_distribution=role,
        traversal_distribution=traversal,
        direction_distribution=direction,
        step_direction_distribution=direction[:,None,:].expand(1,3,3).clone(),
        modifier_weight=modifier,
        step_modifier_weight=step_modifier,
        applicability=torch.ones(1),
        control_distribution=control,
        continuous_state=torch.zeros(1,DIM),
        uncertainty=torch.zeros(1),
    )


def test_binder_global_support_cannot_apply_step0_reliability_to_step1_off_edge() -> None:
    """A global Binder support set is reused by every Executor step.

    When the same relation is executed twice and reliability is ON only for the
    first step, changing reliability on the second-step edge must not alter
    Binder support for that edge. Otherwise the global sparse boundary can
    prune evidence before the step-local Executor gets a chance to honor OFF.
    """
    torch.manual_seed(603)
    binder=FullEnvelopeQSREBinderV1(
        FullEnvelopeBinderConfig(
            semantic_dim=DIM,
            model_dim=DIM,
            num_hidden_states=3,
            edge_metadata_dim=4,
        )
    ).eval()
    # Isolate the direct reliability criterion in Binder support scoring.
    binder.edge_score=_FeatureProbe(5*DIM+1)

    common={
        "query_hidden_states":torch.randn(1,3,5,DIM),
        "query_token_mask":torch.ones(1,5,dtype=torch.bool),
        "field_hidden_states":torch.randn(1,3,3,4,DIM),
        "field_token_mask":torch.ones(1,3,4,dtype=torch.bool),
        "field_state":torch.randn(1,3,DIM),
        "field_valid_mask":torch.ones(1,3,dtype=torch.bool),
        "field_type_index":torch.zeros(1,3,dtype=torch.long),
        "edge_index":torch.tensor([[[0,1],[1,2]]]),
        "edge_relation_index":torch.zeros(1,2,dtype=torch.long),
        "edge_valid_mask":torch.ones(1,2,dtype=torch.bool),
        "edge_recency":torch.full((1,2),0.5),
        "relation_domain_type_mask":torch.ones(1,1,1,dtype=torch.bool),
        "relation_range_type_mask":torch.ones(1,1,1,dtype=torch.bool),
        "relation_symmetric":torch.zeros(1,1,dtype=torch.bool),
        "relation_schema_state":torch.randn(1,1,DIM),
        "operator":_mixed_step_same_relation_operator(),
    }
    with torch.no_grad():
        low=binder(
            edge_reliability=torch.tensor([[0.9,0.1]]),
            **common,
        )
        high=binder(
            edge_reliability=torch.tensor([[0.9,0.9]]),
            **common,
        )

    assert torch.allclose(
        low["support_logits"][:,1],
        high["support_logits"][:,1],
        atol=1.0e-7,
        rtol=0.0,
    )


def _executor_common() -> dict[str, torch.Tensor]:
    torch.manual_seed(602)
    return {
        "field_state": torch.randn(1, 2, DIM),
        "field_metadata": torch.zeros(1, 2, 3),
        "field_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "edge_index": torch.tensor([[[0, 1]]]),
        "edge_relation_index": torch.zeros(1, 1, dtype=torch.long),
        "edge_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "edge_support_weight": torch.ones(1, 1),
        "support_available": torch.ones(1),
        "relation_schema_state": torch.randn(1, 1, DIM),
        "step_relation_schema_state": torch.randn(1, 2, 1, DIM),
        "relation_symmetric": torch.zeros(1, 1, dtype=torch.bool),
        "focus_field_weight": torch.tensor([[1.0, 0.0]]),
    }


def test_executor_modifier_off_blocks_raw_edge_criterion_features() -> None:
    """OFF must neutralize the corresponding raw scalar before edge-state use."""
    common = _executor_common()
    # edge_update input has six D-wide blocks before the four edge scalars.
    scalar_base = 6 * DIM
    cases = (
        (MOD_RELIABILITY, 0, "edge_reliability", 0.1, 0.9),
        (MOD_RECENCY, 1, "edge_recency", 0.1, 0.9),
        (MOD_TEMPORAL_CONSTRAINT, 2, "edge_temporal_match", 0.0, 1.0),
        (MOD_PROVENANCE_CONSTRAINT, 3, "edge_provenance_match", 0.0, 1.0),
    )

    for modifier_index, scalar_offset, key, low, high in cases:
        executor = FullEnvelopeQSREExecutorV1(
            FullEnvelopeExecutorConfig(
                field_dim=DIM,
                model_dim=DIM,
                field_metadata_dim=3,
                edge_metadata_dim=4,
                dropout=0.0,
            )
        ).eval()
        executor.edge_update = _FeatureStateProbe(
            scalar_base + scalar_offset,
            DIM,
        )

        baseline = {
            "edge_reliability": torch.full((1, 1), 0.5),
            "edge_recency": torch.full((1, 1), 0.5),
            "edge_temporal_match": torch.ones(1, 1),
            "edge_provenance_match": torch.ones(1, 1),
        }
        low_case = {name: value.clone() for name, value in baseline.items()}
        high_case = {name: value.clone() for name, value in baseline.items()}
        low_case[key].fill_(low)
        high_case[key].fill_(high)

        with torch.no_grad():
            off_low = executor(
                operator=_operator(),
                **common,
                **low_case,
            )
            off_high = executor(
                operator=_operator(),
                **common,
                **high_case,
            )
            on_low = executor(
                operator=_operator(modifier_index=modifier_index),
                **common,
                **low_case,
            )
            on_high = executor(
                operator=_operator(modifier_index=modifier_index),
                **common,
                **high_case,
            )

        assert torch.allclose(
            off_low["node_state"],
            off_high["node_state"],
            atol=1.0e-7,
            rtol=0.0,
        )
        assert not torch.allclose(
            on_low["node_state"],
            on_high["node_state"],
            atol=1.0e-7,
            rtol=0.0,
        )


class _ForcedModifierAdapter(torch.nn.Module):
    def __init__(self, modifier_index: int | None) -> None:
        super().__init__()
        self.base = SemanticOperatorQSREAdapter()
        self.modifier_index = modifier_index

    def forward(
        self,
        *,
        semantic_operator,
        relation_schema_states,
        factor_opcodes,
    ):
        result = self.base(
            semantic_operator=semantic_operator,
            relation_schema_states=relation_schema_states,
            factor_opcodes=factor_opcodes,
        )
        operator = result["operator"]
        global_modifier = torch.zeros_like(operator.modifier_weight)
        step_modifier = torch.zeros_like(operator.step_modifier_weight)
        if self.modifier_index is not None:
            global_modifier[:, self.modifier_index] = 1.0
            step_modifier[:, :, self.modifier_index] = 1.0
        result["operator"] = replace(
            operator,
            modifier_weight=global_modifier,
            step_modifier_weight=step_modifier,
        )
        return result


class _ForcedMixedStepModifierAdapter(torch.nn.Module):
    """Force a program with reliability ON at step 0 and OFF at step 1."""
    def __init__(self) -> None:
        super().__init__()
        self.base=SemanticOperatorQSREAdapter()

    def forward(
        self,
        *,
        semantic_operator,
        relation_schema_states,
        factor_opcodes,
    ):
        result=self.base(
            semantic_operator=semantic_operator,
            relation_schema_states=relation_schema_states,
            factor_opcodes=factor_opcodes,
        )
        operator=result["operator"]
        global_modifier=torch.zeros_like(operator.modifier_weight)
        global_modifier[:,MOD_RELIABILITY]=1.0
        step_modifier=torch.zeros_like(operator.step_modifier_weight)
        step_modifier[:,0,MOD_RELIABILITY]=1.0
        result["operator"]=replace(
            operator,
            modifier_weight=global_modifier,
            step_modifier_weight=step_modifier,
        )
        return result


def _schema_bank(count: int, seed: int) -> DynamicSemanticSchema:
    torch.manual_seed(seed)
    return DynamicSemanticSchema(
        token_states=torch.randn(count, 3, 4, DIM),
        token_mask=torch.ones(count, 4, dtype=torch.bool),
    )


def _relation_schema() -> DynamicRelationSchema:
    torch.manual_seed(610)
    return DynamicRelationSchema(
        token_states=torch.randn(1, 3, 4, DIM),
        token_mask=torch.ones(1, 4, dtype=torch.bool),
        domain_type_mask=torch.ones(1, 1, dtype=torch.bool),
        range_type_mask=torch.ones(1, 1, dtype=torch.bool),
        symmetric=torch.zeros(1, dtype=torch.bool),
    )


def _factor_bundle():
    schemas = {
        "role": _schema_bank(4, 611),
        "traversal": _schema_bank(3, 612),
        "direction": _schema_bank(3, 613),
        "control": _schema_bank(3, 614),
        "reliability": _schema_bank(2, 615),
        "recency": _schema_bank(2, 616),
        "temporal": _schema_bank(2, 617),
        "provenance": _schema_bank(2, 618),
    }
    opcodes = {
        "role": ["ROLE_SOURCE", "ROLE_TARGET", "ROLE_SYMMETRIC", "ROLE_NONE"],
        "traversal": ["TRAVERSAL_LOCAL", "TRAVERSAL_PATH", "TRAVERSAL_AGGREGATE"],
        "direction": ["DIRECTION_FORWARD", "DIRECTION_REVERSE", "DIRECTION_BIDIRECTIONAL"],
        "control": ["CONTROL_FALLBACK", "CONTROL_RELATIONAL", "CONTROL_DEFER"],
        "reliability": ["MOD_RELIABILITY_OFF", "MOD_RELIABILITY_ON"],
        "recency": ["MOD_RECENCY_OFF", "MOD_RECENCY_ON"],
        "temporal": ["MOD_TEMPORAL_OFF", "MOD_TEMPORAL_ON"],
        "provenance": ["MOD_PROVENANCE_OFF", "MOD_PROVENANCE_ON"],
    }
    return schemas, opcodes


def _stack_inputs() -> dict[str, object]:
    torch.manual_seed(619)
    fields = 2
    edges = 1
    descriptor_banks = {
        "type": _schema_bank(1, 620),
        "provenance": _schema_bank(1, 621),
        "temporal": _schema_bank(1, 622),
    }
    return {
        "query_hidden_states": torch.randn(1, 3, 5, DIM),
        "query_token_mask": torch.ones(1, 5, dtype=torch.bool),
        "field_hidden_states": torch.randn(1, fields, 3, 4, DIM),
        "field_token_mask": torch.ones(1, fields, 4, dtype=torch.bool),
        "field_valid_mask": torch.ones(1, fields, dtype=torch.bool),
        "field_confidence": torch.ones(1, fields) * 0.8,
        "field_missing": torch.zeros(1, fields),
        "field_reliability": torch.tensor([[0.1, 0.9]]),
        "descriptor_banks": descriptor_banks,
        "descriptor_indices": {
            "type": torch.zeros(1, fields, dtype=torch.long),
            "provenance": torch.zeros(1, fields, dtype=torch.long),
            "temporal": torch.zeros(1, fields, dtype=torch.long),
        },
        "field_type_index": torch.zeros(1, fields, dtype=torch.long),
        "field_metadata": torch.zeros(1, fields, 3),
        "edge_index": torch.tensor([[[0, 1]]]),
        "edge_relation_index": torch.zeros(1, edges, dtype=torch.long),
        "edge_valid_mask": torch.ones(1, edges, dtype=torch.bool),
        "edge_metadata": torch.tensor([[[0.1, 0.2, 0.0, 0.0]]]),
        "edge_reliability": torch.tensor([[0.1]]),
        "edge_recency": torch.tensor([[0.2]]),
        "edge_temporal_match": torch.tensor([[0.0]]),
        "edge_provenance_match": torch.tensor([[0.0]]),
        "internal_view_descriptor_states": torch.randn(1, 6, DIM),
        "internal_view_reliability": torch.ones(1, 6),
    }


def _capture_stack_inputs_with_adapter(
    adapter: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    stack=N0FullEnvelopeStackV1(
        N0FullEnvelopeStackConfig(
            semantic_dim=DIM,
            model_dim=DIM,
            num_hidden_states=3,
            num_attention_heads=4,
            structured_layers=1,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    stack.operator_adapter=adapter
    captured: dict[str,torch.Tensor]={}

    def graph_hook(_module,_args,kwargs):
        captured["edge_metadata"]=kwargs["edge_metadata"].detach().clone()

    def evidence_hook(_module,_args,kwargs):
        captured["field_reliability"]=kwargs["field_reliability"].detach().clone()

    graph_handle=stack.evidence_graph.register_forward_pre_hook(
        graph_hook,
        with_kwargs=True,
    )
    evidence_handle=stack.evidence_view.register_forward_pre_hook(
        evidence_hook,
        with_kwargs=True,
    )
    factor_schemas,factor_opcodes=_factor_bundle()
    try:
        with torch.no_grad():
            stack(
                relation_schema=_relation_schema(),
                factor_schemas=factor_schemas,
                factor_opcodes=factor_opcodes,
                max_reasoning_steps=2,
                graph_message_steps=1,
                fusion_refinement_steps=1,
                latent_slot_count=2,
                latent_refinement_steps=1,
                **_stack_inputs(),
            )
    finally:
        graph_handle.remove()
        evidence_handle.remove()
    return captured["edge_metadata"],captured["field_reliability"]


def _capture_stack_modifier_inputs(
    modifier_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    stack = N0FullEnvelopeStackV1(
        N0FullEnvelopeStackConfig(
            semantic_dim=DIM,
            model_dim=DIM,
            num_hidden_states=3,
            num_attention_heads=4,
            structured_layers=1,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            dropout=0.0,
        )
    ).eval()
    stack.operator_adapter = _ForcedModifierAdapter(modifier_index)

    captured: dict[str, torch.Tensor] = {}

    def graph_hook(_module, _args, kwargs):
        captured["edge_metadata"] = kwargs["edge_metadata"].detach().clone()

    def evidence_hook(_module, _args, kwargs):
        captured["field_reliability"] = (
            kwargs["field_reliability"].detach().clone()
        )

    graph_handle = stack.evidence_graph.register_forward_pre_hook(
        graph_hook,
        with_kwargs=True,
    )
    evidence_handle = stack.evidence_view.register_forward_pre_hook(
        evidence_hook,
        with_kwargs=True,
    )
    factor_schemas, factor_opcodes = _factor_bundle()
    try:
        with torch.no_grad():
            stack(
                relation_schema=_relation_schema(),
                factor_schemas=factor_schemas,
                factor_opcodes=factor_opcodes,
                max_reasoning_steps=2,
                graph_message_steps=1,
                fusion_refinement_steps=1,
                latent_slot_count=2,
                latent_refinement_steps=1,
                **_stack_inputs(),
            )
    finally:
        graph_handle.remove()
        evidence_handle.remove()
    return captured["edge_metadata"], captured["field_reliability"]


def test_global_graph_and_evidence_views_are_criterion_neutral() -> None:
    """Global program views cannot safely own step-local evidence criteria."""
    edge_off, field_off = _capture_stack_modifier_inputs(None)
    assert torch.allclose(
        edge_off,
        torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        field_off,
        torch.full_like(field_off, 0.5),
        atol=1.0e-7,
        rtol=0.0,
    )

    for modifier_index in (
        MOD_RELIABILITY,
        MOD_RECENCY,
        MOD_TEMPORAL_CONSTRAINT,
        MOD_PROVENANCE_CONSTRAINT,
    ):
        edge_on,field_on=_capture_stack_modifier_inputs(modifier_index)
        assert torch.allclose(
            edge_on,
            torch.tensor([[[0.5,0.5,1.0,1.0]]]),
            atol=1.0e-7,
            rtol=0.0,
        )
        assert torch.allclose(
            field_on,
            torch.full_like(field_on,0.5),
            atol=1.0e-7,
            rtol=0.0,
        )


def test_global_graph_and_evidence_views_cannot_apply_mixed_step_reliability_globally() -> None:
    """Global views cannot honor both ON and OFF step semantics at once.

    Therefore explicit criterion scalars entering those global views must remain
    criterion-neutral; step-local criterion use belongs to the Executor.
    """
    edge_metadata,field_reliability=_capture_stack_inputs_with_adapter(
        _ForcedMixedStepModifierAdapter()
    )
    assert torch.allclose(
        edge_metadata,
        torch.tensor([[[0.5,0.5,1.0,1.0]]]),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        field_reliability,
        torch.full_like(field_reliability,0.5),
        atol=1.0e-7,
        rtol=0.0,
    )
