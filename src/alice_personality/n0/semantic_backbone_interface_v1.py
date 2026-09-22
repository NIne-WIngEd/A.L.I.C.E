from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor

from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
)


@dataclass(frozen=True)
class SemanticBackboneInterfaceConfig:
    num_hidden_states: int = 17
    semantic_dim: int = 640
    special_token_ids: tuple[int, ...] = (0, 1, 2, 3, 4)

    def validate(self) -> None:
        if self.num_hidden_states <= 0:
            raise ValueError("num_hidden_states must be positive")
        if self.semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        if len(set(int(x) for x in self.special_token_ids)) != len(
            self.special_token_ids
        ):
            raise ValueError("special_token_ids must be unique")


class FullEnvelopeSemanticBackboneInterfaceV1:
    """Gradient-preserving adapter from the shared backbone to full N0 tensors.

    It owns no model parameters. The supplied backbone remains the single shared
    semantic encoder. Every hidden state is preserved; no final-layer-only or
    pooled-only interface is introduced here.
    """

    def __init__(
        self,
        config: SemanticBackboneInterfaceConfig | None = None,
    ) -> None:
        self.config = config or SemanticBackboneInterfaceConfig()
        self.config.validate()

    def _content_mask(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids/attention_mask geometry drift")
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        content = attention_mask.clone()
        for token_id in self.config.special_token_ids:
            content = content & input_ids.ne(int(token_id))
        if bool((content.sum(dim=-1) == 0).any()):
            raise ValueError("every encoded semantic item requires content tokens")
        return content

    def encode_batch(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape drift")

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "semantic backbone did not return hidden_states with output_hidden_states=True"
            )
        if len(hidden_states) != self.config.num_hidden_states:
            raise ValueError(
                "semantic hidden-state depth drift: "
                f"expected {self.config.num_hidden_states}, got {len(hidden_states)}"
            )
        first_shape = hidden_states[0].shape
        if (
            len(first_shape) != 3
            or first_shape[0] != input_ids.size(0)
            or first_shape[1] != input_ids.size(1)
            or first_shape[2] != self.config.semantic_dim
        ):
            raise ValueError("semantic hidden-state geometry drift")
        for index, state in enumerate(hidden_states):
            if state.shape != first_shape:
                raise ValueError(
                    f"semantic hidden-state shape drift at layer {index}"
                )
        stack = torch.stack(tuple(hidden_states), dim=1)
        content_mask = self._content_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return {
            "hidden_states": stack,
            "content_mask": content_mask,
            "attention_mask": attention_mask.bool(),
        }

    def encode_semantic_bank(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> DynamicSemanticSchema:
        encoded = self.encode_batch(
            backbone=backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        schema = DynamicSemanticSchema(
            token_states=encoded["hidden_states"],
            token_mask=encoded["content_mask"],
        )
        schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        return schema

    def encode_relation_bank(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
        domain_type_mask: Tensor,
        range_type_mask: Tensor,
        symmetric: Tensor,
    ) -> DynamicRelationSchema:
        encoded = self.encode_batch(
            backbone=backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        schema = DynamicRelationSchema(
            token_states=encoded["hidden_states"],
            token_mask=encoded["content_mask"],
            domain_type_mask=domain_type_mask,
            range_type_mask=range_type_mask,
            symmetric=symmetric,
        )
        schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        return schema

    def encode_fields(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
        field_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if input_ids.ndim != 3:
            raise ValueError("field input_ids must be [B,F,T]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("field attention_mask shape drift")
        batch, fields, tokens = input_ids.shape
        if field_valid_mask.shape != (batch, fields):
            raise ValueError("field_valid_mask must be [B,F]")
        if field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool")
        if bool((field_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires one valid field")

        flat_ids = input_ids.reshape(batch * fields, tokens)
        flat_attention = attention_mask.reshape(batch * fields, tokens).bool()

        # Padded invalid fields may have no content. Supply one temporary
        # attended token for backbone mechanics, then zero/mask the resulting
        # semantic tensor before exposing it downstream.
        flat_valid = field_valid_mask.reshape(batch * fields)
        safe_attention = flat_attention.clone()
        if bool((~flat_valid).any()):
            safe_attention[~flat_valid, 0] = True

        outputs = backbone(
            input_ids=flat_ids,
            attention_mask=safe_attention,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) != self.config.num_hidden_states:
            raise RuntimeError("field backbone hidden-state depth drift")
        stack = torch.stack(tuple(hidden_states), dim=1)
        if stack.shape != (
            batch * fields,
            self.config.num_hidden_states,
            tokens,
            self.config.semantic_dim,
        ):
            raise ValueError("field backbone hidden-state geometry drift")
        stack = stack.reshape(
            batch,
            fields,
            self.config.num_hidden_states,
            tokens,
            self.config.semantic_dim,
        )
        stack = stack * field_valid_mask[:, :, None, None, None].to(
            stack.dtype
        )

        content = safe_attention.clone()
        for token_id in self.config.special_token_ids:
            content = content & flat_ids.ne(int(token_id))
        content = content & flat_valid[:, None]
        content = content.reshape(batch, fields, tokens)
        if bool(
            (
                field_valid_mask
                & content.sum(dim=-1).eq(0)
            ).any()
        ):
            raise ValueError("valid field has no semantic content tokens")

        return {
            "field_hidden_states": stack,
            "field_token_mask": content,
            "field_valid_mask": field_valid_mask,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "owned_parameters": 0,
            "shared_backbone_reference_only": True,
            "all_hidden_states_preserved": True,
            "final_layer_only": False,
            "pooled_only": False,
            "semantic_backbone_gradient_detached": False,
            "runtime_schema_candidate_ceiling": None,
            "runtime_field_count_ceiling": None,
        }
