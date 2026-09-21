from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from alice_personality.n0.qsre_production_core import QSREProductionOperatorState
from alice_personality.n0.semantic_operator_foundation import SemanticOperatorState


ROLE_OPCODES = ("ROLE_SOURCE", "ROLE_TARGET", "ROLE_SYMMETRIC", "ROLE_NONE")
TRAVERSAL_OPCODES = ("TRAVERSAL_LOCAL", "TRAVERSAL_PATH", "TRAVERSAL_AGGREGATE")
DIRECTION_OPCODES = ("DIRECTION_FORWARD", "DIRECTION_REVERSE", "DIRECTION_BIDIRECTIONAL")
CONTROL_OPCODES = ("CONTROL_FALLBACK", "CONTROL_RELATIONAL", "CONTROL_DEFER")
MODIFIER_OPCODES = (
    ("MOD_RELIABILITY_OFF", "MOD_RELIABILITY_ON"),
    ("MOD_RECENCY_OFF", "MOD_RECENCY_ON"),
    ("MOD_TEMPORAL_OFF", "MOD_TEMPORAL_ON"),
    ("MOD_PROVENANCE_OFF", "MOD_PROVENANCE_ON"),
)


class SemanticOperatorQSREAdapter(nn.Module):
    """Parameter-free structural adapter from semantic schemas to QSRE opcodes.

    Natural-language factor descriptions own semantic selection. Structural
    opcodes only say what an already-selected semantic factor *does* in the
    executor. They are runtime metadata, not learned identity embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _opcode_mass(
        factor_distributions: Mapping[str, Tensor],
        factor_opcodes: Mapping[str, Sequence[str]],
    ) -> dict[str, Tensor]:
        if set(factor_distributions) != set(factor_opcodes):
            raise ValueError("factor distributions/opcode metadata names must match")
        result: dict[str, Tensor] = {}
        for name, probability in factor_distributions.items():
            opcodes = list(factor_opcodes[name])
            if probability.ndim != 2 or probability.size(1) != len(opcodes):
                raise ValueError(f"factor opcode cardinality drift for {name!r}")
            for index, opcode in enumerate(opcodes):
                value = probability[:, index]
                result[opcode] = result.get(opcode, torch.zeros_like(value)) + value
        return result

    @staticmethod
    def _canonical_distribution(
        mass: Mapping[str, Tensor],
        opcodes: Sequence[str],
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        rows = []
        for opcode in opcodes:
            if opcode not in mass:
                raise ValueError(f"required structural opcode missing: {opcode}")
            rows.append(mass[opcode])
        value = torch.stack(rows, dim=-1).to(device=device, dtype=dtype)
        return value / value.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    @staticmethod
    def _modifier_probability(
        mass: Mapping[str, Tensor],
        off_opcode: str,
        on_opcode: str,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        off = mass.get(off_opcode)
        on = mass.get(on_opcode)
        if off is None and on is None:
            return torch.zeros(batch, device=device, dtype=dtype)
        if off is None:
            off = torch.zeros_like(on)
        if on is None:
            on = torch.zeros_like(off)
        denom = (off + on).clamp_min(1.0e-12)
        return (on / denom).to(device=device, dtype=dtype)

    def forward(
        self,
        *,
        semantic_operator: SemanticOperatorState,
        relation_schema_states: Tensor,
        factor_opcodes: Mapping[str, Sequence[str]],
    ) -> dict[str, Tensor | QSREProductionOperatorState]:
        relation = semantic_operator.relation_distribution
        batch, steps, relations = relation.shape
        if relation_schema_states.ndim != 4:
            raise ValueError("relation_schema_states must be [B,S,R,D]")
        if relation_schema_states.shape[:3] != (batch, steps, relations):
            raise ValueError("relation schema-state geometry drift")

        mass = self._opcode_mass(
            semantic_operator.factor_distributions,
            factor_opcodes,
        )
        device = relation.device
        dtype = relation.dtype
        role = self._canonical_distribution(
            mass, ROLE_OPCODES, batch=batch, device=device, dtype=dtype
        )
        traversal = self._canonical_distribution(
            mass, TRAVERSAL_OPCODES, batch=batch, device=device, dtype=dtype
        )
        direction = self._canonical_distribution(
            mass, DIRECTION_OPCODES, batch=batch, device=device, dtype=dtype
        )
        control = self._canonical_distribution(
            mass, CONTROL_OPCODES, batch=batch, device=device, dtype=dtype
        )
        modifier = torch.stack(
            [
                self._modifier_probability(
                    mass, off, on, batch=batch, device=device, dtype=dtype
                )
                for off, on in MODIFIER_OPCODES
            ],
            dim=-1,
        )

        step_mass = semantic_operator.relation_step_mass
        normalizer = step_mass.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        relation_schema_state = (
            relation_schema_states
            * step_mass[:, :, None, None].to(relation_schema_states.dtype)
        ).sum(dim=1) / normalizer[:, :, None].to(relation_schema_states.dtype)

        operator = QSREProductionOperatorState(
            relation_distribution=semantic_operator.relation_distribution,
            relation_step_mass=semantic_operator.relation_step_mass,
            stop_probability=semantic_operator.stop_probability,
            unknown_probability=semantic_operator.unknown_probability,
            role_distribution=role,
            traversal_distribution=traversal,
            direction_distribution=direction,
            modifier_weight=modifier,
            applicability=semantic_operator.applicability,
            control_distribution=control,
            continuous_state=semantic_operator.continuous_state,
            uncertainty=semantic_operator.uncertainty,
        )
        operator.validate(
            relation_count=relations,
            model_dim=semantic_operator.continuous_state.size(-1),
        )
        return {
            "operator": operator,
            "relation_schema_state": relation_schema_state,
        }

    def parameter_report(self) -> dict[str, int | bool | None]:
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "factor_identity_parameters": 0,
            "relation_identity_parameters": 0,
            "structural_opcodes_are_runtime_metadata": True,
            "semantic_selection_owned_by_runtime_schema": True,
            "relation_count_ceiling": None,
            "factor_candidate_count_ceiling": None,
        }
