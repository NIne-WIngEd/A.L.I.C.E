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


def test_binder_modifier_off_blocks_raw_reliability_and_recency_shortcuts() -> None:
    """A disabled criterion may not remain directly readable by Binder."""
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
        assert not torch.allclose(
            enabled["support_logits"][:, 0],
            enabled["support_logits"][:, 1],
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
