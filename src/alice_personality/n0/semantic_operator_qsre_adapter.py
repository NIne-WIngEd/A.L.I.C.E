from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from alice_personality.n0.full_envelope_structural_types import FullEnvelopeOperatorState
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

SUPPORTED_STRUCTURAL_OPCODES = frozenset(
    ROLE_OPCODES
    + TRAVERSAL_OPCODES
    + DIRECTION_OPCODES
    + CONTROL_OPCODES
    + tuple(opcode for pair in MODIFIER_OPCODES for opcode in pair)
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
        unknown = set(factor_opcodes) - set(factor_distributions)
        if unknown:
            raise ValueError(
                "factor opcode metadata supplied for unknown semantic banks: "
                + repr(sorted(unknown))
            )
        result: dict[str, Tensor] = {}
        for name, opcodes_raw in factor_opcodes.items():
            probability = factor_distributions[name]
            opcodes = list(opcodes_raw)
            if probability.ndim != 2 or probability.size(1) != len(opcodes):
                raise ValueError(f"factor opcode cardinality drift for {name!r}")
            for index, opcode in enumerate(opcodes):
                if opcode not in SUPPORTED_STRUCTURAL_OPCODES:
                    raise ValueError(
                        f"unsupported structural opcode {opcode!r}; "
                        "new executable primitives require an explicit architecture version"
                    )
                value = probability[:, index]
                result[opcode] = result.get(opcode, torch.zeros_like(value)) + value
        return result

    @staticmethod
    def _step_opcode_mass(
        factor_distributions: Mapping[str, Tensor],
        factor_opcodes: Mapping[str, Sequence[str]],
    ) -> dict[str, Tensor]:
        unknown = set(factor_opcodes) - set(factor_distributions)
        if unknown:
            raise ValueError(
                "step factor opcode metadata supplied for unknown semantic banks: "
                + repr(sorted(unknown))
            )
        result: dict[str, Tensor] = {}
        for name, opcodes_raw in factor_opcodes.items():
            probability = factor_distributions[name]
            opcodes = list(opcodes_raw)
            if probability.ndim != 3 or probability.size(-1) != len(opcodes):
                raise ValueError(
                    f"step factor opcode cardinality drift for {name!r}"
                )
            for index, opcode in enumerate(opcodes):
                if opcode not in SUPPORTED_STRUCTURAL_OPCODES:
                    raise ValueError(
                        f"unsupported structural opcode {opcode!r}; "
                        "new executable primitives require an explicit architecture version"
                    )
                value = probability[:, :, index]
                result[opcode] = result.get(
                    opcode,
                    torch.zeros_like(value),
                ) + value
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

    @staticmethod
    def _canonical_step_distribution(
        mass: Mapping[str, Tensor],
        opcodes: Sequence[str],
        *,
        batch: int,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        rows = []
        for opcode in opcodes:
            if opcode not in mass:
                raise ValueError(f"required structural opcode missing: {opcode}")
            rows.append(mass[opcode])
        value = torch.stack(rows, dim=-1).to(device=device, dtype=dtype)
        if value.shape[:2] != (batch, steps):
            raise ValueError("step structural opcode geometry drift")
        return value / value.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    @staticmethod
    def _step_modifier_probability(
        mass: Mapping[str, Tensor],
        off_opcode: str,
        on_opcode: str,
        *,
        batch: int,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        off = mass.get(off_opcode)
        on = mass.get(on_opcode)
        if off is None and on is None:
            return torch.zeros(batch, steps, device=device, dtype=dtype)
        if off is None:
            off = torch.zeros_like(on)
        if on is None:
            on = torch.zeros_like(off)
        denom = (off + on).clamp_min(1.0e-12)
        value = (on / denom).to(device=device, dtype=dtype)
        if value.shape != (batch, steps):
            raise ValueError("step modifier geometry drift")
        return value

    def forward(
        self,
        *,
        semantic_operator: SemanticOperatorState,
        relation_schema_states: Tensor,
        factor_opcodes: Mapping[str, Sequence[str]],
    ) -> dict[str, Tensor | FullEnvelopeOperatorState]:
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
        step_mass_by_opcode = self._step_opcode_mass(
            semantic_operator.step_factor_distributions,
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
        step_direction = self._canonical_step_distribution(
            step_mass_by_opcode,
            DIRECTION_OPCODES,
            batch=batch,
            steps=steps,
            device=device,
            dtype=dtype,
        )
        step_modifier = torch.stack(
            [
                self._step_modifier_probability(
                    step_mass_by_opcode,
                    off,
                    on,
                    batch=batch,
                    steps=steps,
                    device=device,
                    dtype=dtype,
                )
                for off, on in MODIFIER_OPCODES
            ],
            dim=-1,
        )

        step_mass = semantic_operator.relation_step_mass
        relation_assignment_mass = (
            step_mass[:, :, None]
            * semantic_operator.relation_distribution
        )
        relation_normalizer = relation_assignment_mass.sum(
            dim=1
        ).clamp_min(1.0e-6)
        relation_schema_state = (
            relation_schema_states
            * relation_assignment_mass[:, :, :, None].to(
                relation_schema_states.dtype
            )
        ).sum(dim=1) / relation_normalizer[:, :, None].to(
            relation_schema_states.dtype
        )

        operator = FullEnvelopeOperatorState(
            relation_distribution=semantic_operator.relation_distribution,
            relation_step_mass=semantic_operator.relation_step_mass,
            stop_probability=semantic_operator.stop_probability,
            unknown_probability=semantic_operator.unknown_probability,
            truncation_probability=semantic_operator.truncation_probability,
            role_distribution=role,
            traversal_distribution=traversal,
            direction_distribution=direction,
            step_direction_distribution=step_direction,
            modifier_weight=modifier,
            step_modifier_weight=step_modifier,
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
            "step_relation_schema_state": relation_schema_states,
        }

    def parameter_report(self) -> dict[str, int | bool | None]:
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "factor_identity_parameters": 0,
            "relation_identity_parameters": 0,
            "structural_opcodes_are_runtime_metadata": True,
            "semantic_selection_owned_by_runtime_schema": True,
            "structural_factor_mapping_may_be_subset_of_semantic_banks": True,
            "semantic_only_factor_banks_supported": True,
            "runtime_semantic_factor_bank_ceiling": None,
            "step_conditioned_direction_and_modifiers": True,
            "program_global_role_semantics": True,
            "program_global_traversal_semantics": True,
            "program_global_control_semantics": True,
            "step_local_structural_semantics_are_direction_and_modifiers": True,
            "step_factor_semantic_context_may_include_program_global_banks": True,
            "program_relation_state_probability_weighted": True,
            "step_conditioned_relation_schema_state_preserved": True,
            "program_truncation_preserved": True,
            "relation_count_ceiling": None,
            "factor_candidate_count_ceiling": None,
        }
